"""Versionable experiment specifications, manifests, and dry-run matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.inference.cache import inspect_model_cache
from src.inference.config import load_model_config


class ExperimentKind(str, Enum):
    ZERO_SHOT = "zero_shot"
    LORA = "lora"
    QLORA = "qlora"
    QUANTIZATION = "quantization"
    SPECIALIST = "specialist"
    FUSION = "fusion"


class RunStatus(str, Enum):
    NOT_RUN = "not_run"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class EvidenceStatus(str, Enum):
    NOT_RUN = "not_run"
    RECORDED = "recorded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class EvidenceValue:
    status: EvidenceStatus
    value: Any = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is EvidenceStatus.RECORDED and self.value is None:
            raise ValueError("recorded evidence requires a value")
        if self.status is not EvidenceStatus.RECORDED and self.value is not None:
            raise ValueError("not-run/unavailable evidence cannot contain a value")
        if self.status is EvidenceStatus.UNAVAILABLE and not (self.reason or "").strip():
            raise ValueError("unavailable evidence requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "value": self.value, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    kind: ExperimentKind
    base_model_id: str
    model_revision: str
    base_config_path: str
    base_config_fingerprint: str
    data_version: str
    seed: int
    quantization_bits: int
    input_resolution: int
    adapter_method: str | None = None
    adapter_rank: int | None = None
    module_strategy: str | None = None
    include_hard_negatives: bool | None = None
    specialists: tuple[str, ...] = ()
    parameters: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    status: RunStatus = RunStatus.NOT_RUN

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "base_model_id",
            "model_revision",
            "base_config_path",
            "base_config_fingerprint",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.quantization_bits not in {4, 8, 16}:
            raise ValueError("quantization_bits must be 4, 8, or 16")
        if self.input_resolution <= 0:
            raise ValueError("input_resolution must be positive")
        if self.kind in {ExperimentKind.LORA, ExperimentKind.QLORA}:
            if self.adapter_method != self.kind.value or not self.adapter_rank:
                raise ValueError(f"{self.kind.value} requires matching method and positive rank")
            if self.module_strategy not in {"language_only", "vision_language"}:
                raise ValueError("adapter experiment requires a supported module_strategy")
        elif self.adapter_method is not None or self.adapter_rank is not None:
            raise ValueError("non-adapter experiment cannot set adapter method or rank")
        if self.kind is ExperimentKind.QLORA and self.quantization_bits != 4:
            raise ValueError("QLoRA experiment requires 4-bit base weights")
        if self.kind is ExperimentKind.SPECIALIST and len(self.specialists) != 1:
            raise ValueError("specialist baseline requires exactly one specialist")
        if self.kind is ExperimentKind.FUSION and not self.specialists:
            raise ValueError("fusion experiment requires at least one specialist")

    @property
    def fingerprint(self) -> str:
        payload = self.to_dict()
        payload.pop("base_config_path")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "kind": self.kind.value,
            "base_model_id": self.base_model_id,
            "model_revision": self.model_revision,
            "base_config_path": self.base_config_path,
            "base_config_fingerprint": self.base_config_fingerprint,
            "data_version": self.data_version,
            "seed": self.seed,
            "quantization_bits": self.quantization_bits,
            "input_resolution": self.input_resolution,
            "adapter_method": self.adapter_method,
            "adapter_rank": self.adapter_rank,
            "module_strategy": self.module_strategy,
            "include_hard_negatives": self.include_hard_negatives,
            "specialists": list(self.specialists),
            "parameters": dict(self.parameters),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    experiment_id: str
    spec_fingerprint: str
    status: RunStatus
    code_version: EvidenceValue
    data_version: EvidenceValue
    started_at: EvidenceValue
    ended_at: EvidenceValue
    adapter_hash: EvidenceValue
    metrics: Mapping[str, EvidenceValue]
    resources: Mapping[str, EvidenceValue]
    artifacts: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        for field_name in ("run_id", "experiment_id", "spec_fingerprint"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.status is RunStatus.NOT_RUN:
            if self.run_id != "not_run":
                raise ValueError("not-run manifest requires run_id=not_run")
            execution_evidence = (self.started_at, self.ended_at, self.adapter_hash)
            if any(item.status is EvidenceStatus.RECORDED for item in execution_evidence):
                raise ValueError("not-run manifest cannot contain execution evidence")
            measured = tuple(self.metrics.values()) + tuple(self.resources.values())
            if any(item.status is EvidenceStatus.RECORDED for item in measured):
                raise ValueError("not-run manifest cannot contain measured evidence")
        if self.status is RunStatus.COMPLETED and any(
            item.status is not EvidenceStatus.RECORDED
            for item in (self.started_at, self.ended_at)
        ):
            raise ValueError("completed manifest requires recorded start and end times")

    @classmethod
    def not_run(cls, spec: ExperimentSpec) -> "RunManifest":
        marker = EvidenceValue(EvidenceStatus.NOT_RUN)
        data_version = (
            marker
            if spec.data_version in {"not_run", "unavailable", "not_set"}
            else EvidenceValue(EvidenceStatus.RECORDED, spec.data_version)
        )
        return cls(
            run_id="not_run",
            experiment_id=spec.experiment_id,
            spec_fingerprint=spec.fingerprint,
            status=RunStatus.NOT_RUN,
            code_version=marker,
            data_version=data_version,
            started_at=marker,
            ended_at=marker,
            adapter_hash=marker,
            metrics={
                name: marker
                for name in (
                    "macro_f1",
                    "acc_at_iou_0_5",
                    "json_schema_validity_rate",
                    "hard_negative_false_positive_rate",
                    "evidence_conclusion_consistency_rate",
                )
            },
            resources={
                name: marker
                for name in ("p50_latency_ms", "p95_latency_ms", "peak_memory_mb")
            },
            artifacts={
                name: marker
                for name in ("predictions_jsonl", "evaluation_report", "adapter_path")
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "spec_fingerprint": self.spec_fingerprint,
            "status": self.status.value,
            "code_version": self.code_version.to_dict(),
            "data_version": self.data_version.to_dict(),
            "execution": {
                "started_at": self.started_at.to_dict(),
                "ended_at": self.ended_at.to_dict(),
            },
            "adapter_hash": self.adapter_hash.to_dict(),
            "metrics": {key: value.to_dict() for key, value in self.metrics.items()},
            "resources": {key: value.to_dict() for key, value in self.resources.items()},
            "artifacts": {key: value.to_dict() for key, value in self.artifacts.items()},
        }


@dataclass(frozen=True, slots=True)
class DryRunReport:
    valid: bool
    runnable: bool
    experiment_count: int
    counts_by_kind: Mapping[str, int]
    errors: tuple[str, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "runnable": self.runnable,
            "experiment_count": self.experiment_count,
            "counts_by_kind": dict(self.counts_by_kind),
            "errors": list(self.errors),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


def generate_default_matrix(
    base_config_path: str | Path,
    *,
    data_version: str = "unavailable",
    seed: int = 20260808,
) -> tuple[ExperimentSpec, ...]:
    config = load_model_config(base_config_path)
    common = {
        "base_model_id": config.base_model_id,
        "model_revision": config.revision,
        "base_config_path": str(base_config_path),
        "base_config_fingerprint": config.fingerprint,
        "data_version": data_version,
        "seed": seed,
    }
    specs: list[ExperimentSpec] = [
        ExperimentSpec(
            "zero-shot-q4-r1024",
            ExperimentKind.ZERO_SHOT,
            quantization_bits=4,
            input_resolution=1024,
            **common,
        )
    ]
    for kind in (ExperimentKind.LORA, ExperimentKind.QLORA):
        for rank in (8, 16):
            for strategy in ("language_only", "vision_language"):
                hard_negative_values = (False, True) if kind is ExperimentKind.QLORA else (True,)
                for hard_negatives in hard_negative_values:
                    suffix = (
                        f"-hardneg-{int(hard_negatives)}"
                        if kind is ExperimentKind.QLORA
                        else ""
                    )
                    specs.append(
                        ExperimentSpec(
                            f"{kind.value}-r{rank}-{strategy.replace('_', '-')}{suffix}",
                            kind,
                            quantization_bits=4 if kind is ExperimentKind.QLORA else 16,
                            input_resolution=1024,
                            adapter_method=kind.value,
                            adapter_rank=rank,
                            module_strategy=strategy,
                            include_hard_negatives=hard_negatives,
                            **common,
                        )
                    )
    for bits in (4, 8, 16):
        for resolution in (1024, 2048):
            specs.append(
                ExperimentSpec(
                    f"quant-q{bits}-r{resolution}",
                    ExperimentKind.QUANTIZATION,
                    quantization_bits=bits,
                    input_resolution=resolution,
                    **common,
                )
            )
    for specialist in ("florence2", "rf_detr"):
        specs.append(
            ExperimentSpec(
                f"specialist-{specialist.replace('_', '-')}",
                ExperimentKind.SPECIALIST,
                quantization_bits=16,
                input_resolution=1024,
                specialists=(specialist,),
                **common,
            )
        )
        specs.append(
            ExperimentSpec(
                f"fusion-vlm-{specialist.replace('_', '-')}",
                ExperimentKind.FUSION,
                quantization_bits=4,
                input_resolution=1024,
                specialists=(specialist,),
                **common,
            )
        )
    specs.append(
        ExperimentSpec(
            "fusion-vlm-florence2-rf-detr",
            ExperimentKind.FUSION,
            quantization_bits=4,
            input_resolution=1024,
            specialists=("florence2", "rf_detr"),
            **common,
        )
    )
    return tuple(specs)


def validate_experiment_matrix(
    specs: Sequence[ExperimentSpec],
    *,
    cache_root: str | Path | None = None,
    available_specialists: Sequence[str] = (),
    available_quantization_bits: Sequence[int] = (),
    available_adaptations: Sequence[str] = (),
) -> DryRunReport:
    errors: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    identifiers = [spec.experiment_id for spec in specs]
    duplicates = sorted(
        identifier
        for identifier, count in Counter(identifiers).items()
        if count > 1
    )
    if duplicates:
        errors.append(f"duplicate experiment_id values: {duplicates}")
    kinds = Counter(spec.kind.value for spec in specs)
    missing_kinds = sorted(kind.value for kind in ExperimentKind if not kinds[kind.value])
    if missing_kinds:
        errors.append(f"matrix is missing required experiment kinds: {missing_kinds}")
    missing_configs = sorted(
        {spec.base_config_path for spec in specs if not Path(spec.base_config_path).is_file()}
    )
    if missing_configs:
        blockers.append(f"base config files unavailable: {missing_configs}")
    configured_quantization_bits: set[int] = set()
    for config_path in sorted(set(spec.base_config_path for spec in specs) - set(missing_configs)):
        try:
            config = load_model_config(config_path)
            cache = inspect_model_cache(config, cache_root=cache_root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid base config {config_path}: {exc}")
            continue
        configured_quantization_bits.add(config.quantization.bits)
        if not cache.complete:
            blockers.append(
                "pinned model snapshot unavailable: "
                f"{config.model_id}@{config.revision} ({cache.reason})"
            )
    unavailable_data = sorted(
        {
            spec.data_version
            for spec in specs
            if spec.data_version in {"not_run", "unavailable", "not_set"}
        }
    )
    if unavailable_data:
        blockers.append("frozen data version is unavailable")
    requested_specialists = {
        specialist for spec in specs for specialist in spec.specialists
    }
    unavailable_specialists = sorted(requested_specialists - set(available_specialists))
    if unavailable_specialists:
        blockers.append(f"specialist implementations unavailable: {unavailable_specialists}")
    requested_bits = {spec.quantization_bits for spec in specs}
    available_bits = configured_quantization_bits | set(available_quantization_bits)
    unavailable_bits = sorted(requested_bits - available_bits)
    if unavailable_bits:
        blockers.append(f"quantization artifacts unavailable: {unavailable_bits}")
    requested_adaptations = {
        spec.adapter_method for spec in specs if spec.adapter_method is not None
    }
    unavailable_adaptations = sorted(
        requested_adaptations - set(available_adaptations)
    )
    if unavailable_adaptations:
        blockers.append(
            f"adaptation implementations unavailable: {unavailable_adaptations}"
        )
    if any(spec.status is not RunStatus.NOT_RUN for spec in specs):
        warnings.append("dry-run matrix contains a status other than not_run")
    return DryRunReport(
        valid=not errors,
        runnable=not errors and not blockers,
        experiment_count=len(specs),
        counts_by_kind=dict(sorted(kinds.items())),
        errors=tuple(errors),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def matrix_payload(
    specs: Sequence[ExperimentSpec],
    *,
    cache_root: str | Path | None = None,
    available_specialists: Sequence[str] = (),
    available_quantization_bits: Sequence[int] = (),
    available_adaptations: Sequence[str] = (),
) -> dict[str, Any]:
    report = validate_experiment_matrix(
        specs,
        cache_root=cache_root,
        available_specialists=available_specialists,
        available_quantization_bits=available_quantization_bits,
        available_adaptations=available_adaptations,
    )
    return {
        "schema_version": "1.0",
        "status": RunStatus.NOT_RUN.value,
        "dry_run": report.to_dict(),
        "experiments": [
            {
                "spec": spec.to_dict() | {"fingerprint": spec.fingerprint},
                "manifest": RunManifest.not_run(spec).to_dict(),
            }
            for spec in specs
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-version", default="unavailable")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--available-specialist",
        action="append",
        choices=("florence2", "rf_detr"),
        default=[],
    )
    parser.add_argument(
        "--available-quantization-bit",
        action="append",
        type=int,
        choices=(4, 8, 16),
        default=[],
    )
    parser.add_argument(
        "--available-adaptation",
        action="append",
        choices=("lora", "qlora"),
        default=[],
    )
    parser.add_argument("--require-runnable", action="store_true")
    arguments = parser.parse_args(argv)
    payload = matrix_payload(
        generate_default_matrix(
            arguments.config,
            data_version=arguments.data_version,
        ),
        cache_root=arguments.cache_root,
        available_specialists=arguments.available_specialist,
        available_quantization_bits=arguments.available_quantization_bit,
        available_adaptations=arguments.available_adaptation,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if arguments.require_runnable and not payload["dry_run"]["runnable"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
