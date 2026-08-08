"""Import domain JSONL into the canonical dataset, manifest, and data card."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data_card import generate_data_card
from .license_policy import LicensePolicy
from .pipeline import prepare_dataset
from .schema import DataSample, SchemaError
from .statistics import compute_dataset_statistics
from .validation import validate_dataset


@dataclass(frozen=True)
class RejectedImport:
    line_number: int
    sample_id: str | None
    code: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DomainImportResult:
    samples: tuple[DataSample, ...]
    rejected: tuple[RejectedImport, ...]


def _stable_sample_id(domain: str, value: Mapping[str, Any]) -> str:
    normalized_domain = re.sub(r"[^a-z0-9_]+", "_", domain.lower()).strip("_")
    if not normalized_domain:
        raise ValueError("domain must contain letters or digits")
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{normalized_domain}_{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _canonical_payload(value: Mapping[str, Any], domain: str) -> dict[str, Any]:
    payload = dict(value)
    if not payload.get("sample_id"):
        payload["sample_id"] = _stable_sample_id(domain, value)
    if "response" not in payload:
        payload["response"] = {
            "result": payload.pop("result", None),
            "objects": payload.pop("objects", []),
            "reason": payload.pop("reason", "domain annotation"),
            "uncertain": payload.pop("uncertain", False),
        }
    difficulty = payload.get("difficulty", [])
    if payload.pop("hard_negative", False) and "hard_negative" not in difficulty:
        difficulty = [*difficulty, "hard_negative"]
    payload["difficulty"] = difficulty
    explicit_slices = payload.pop("slices", None)
    if explicit_slices is not None:
        metadata = dict(payload.get("metadata", {}))
        metadata["slices"] = explicit_slices
        payload["metadata"] = metadata
    return payload


def import_domain_samples(
    input_path: str | Path,
    dataset_root: str | Path,
    policy: LicensePolicy,
    *,
    domain: str,
) -> DomainImportResult:
    accepted = []
    rejected = []
    seen_ids: set[str] = set()
    with Path(input_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            sample_id = None
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise SchemaError("input row must be a JSON object")
                payload = _canonical_payload(value, domain)
                sample_id = payload.get("sample_id")
                sample = DataSample.from_dict(payload)
                if sample.sample_id in seen_ids:
                    raise SchemaError("DUPLICATE_SAMPLE_ID: sample_id is not unique")
                decision = policy.decide(sample.license)
                if not decision.allowed:
                    raise SchemaError(f"{decision.code}: {decision.message}")
                validation = validate_dataset([sample], dataset_root)
                if not validation.valid:
                    issue = validation.errors[0]
                    raise SchemaError(f"{issue.code}: {issue.message}")
                seen_ids.add(sample.sample_id)
                accepted.append(sample)
            except (json.JSONDecodeError, SchemaError, TypeError, ValueError) as exc:
                message = str(exc)
                code = message.partition(":")[0] if ":" in message else "INVALID_SAMPLE"
                rejected.append(RejectedImport(line_number, str(sample_id) if sample_id else None, code, message))
    return DomainImportResult(tuple(accepted), tuple(rejected))


def _atomic_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _jsonl(values: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import visual-compliance domain samples")
    parser.add_argument("--input", required=True, help="source JSONL")
    parser.add_argument("--dataset-root", required=True, help="root containing relative image paths")
    parser.add_argument("--output", required=True, help="canonical sample JSONL")
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--data-card-output", required=True)
    parser.add_argument("--rejects-output", required=True)
    parser.add_argument("--license-policy", required=True)
    parser.add_argument("--domain", default="compliance")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hash-distance", type=int, default=5)
    parser.add_argument("--fixture-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = import_domain_samples(
        arguments.input,
        arguments.dataset_root,
        LicensePolicy.load(arguments.license_policy),
        domain=arguments.domain,
    )
    _atomic_text(arguments.rejects_output, _jsonl([item.to_dict() for item in result.rejected]))
    initial_summary = {
        "input_rows": len(result.samples) + len(result.rejected),
        "accepted_rows": len(result.samples),
        "rejected_rows": len(result.rejected),
        "rejection_codes": dict(
            sorted(
                {
                    code: sum(item.code == code for item in result.rejected)
                    for code in {item.code for item in result.rejected}
                }.items()
            )
        ),
    }
    if result.rejected and not arguments.allow_partial:
        _atomic_text(arguments.summary_output, json.dumps(initial_summary, indent=2, sort_keys=True) + "\n")
        return 2
    if not result.samples:
        raise ValueError("no samples passed import validation")

    prepared = prepare_dataset(
        result.samples,
        arguments.dataset_root,
        dataset_version=arguments.dataset_version,
        created_at=arguments.created_at,
        seed=arguments.seed,
        max_hash_distance=arguments.max_hash_distance,
        frozen_test=True,
        split_strategy_metadata={"import_domain": arguments.domain},
    )
    statistics = compute_dataset_statistics(result.samples, prepared.assignments)
    summary = initial_summary | {
        "dataset": statistics,
        "near_duplicate_group_count": len(prepared.duplicate_groups),
        "near_duplicate_leakage_rate": prepared.leakage.near_duplicate_leakage_rate,
    }
    _atomic_text(arguments.output, _jsonl([sample.to_dict() for sample in result.samples]))
    prepared.manifest.write_atomic(arguments.manifest_output)
    _atomic_text(arguments.summary_output, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(
        arguments.data_card_output,
        generate_data_card(
            prepared.manifest,
            statistics,
            title=f"{arguments.domain.title()} Dataset Card",
            fixture_only=arguments.fixture_only,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
