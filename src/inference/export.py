"""Atomic batch export to the evaluator's model-result JSONL contract."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .base import ModelAdapter
from .contracts import ModelRequest
from .serialization import model_result_to_dict


@dataclass(frozen=True, slots=True)
class BatchPredictionSample:
    sample_id: str
    request: ModelRequest

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")


@dataclass(frozen=True, slots=True)
class PredictionExportSummary:
    destination: str
    total: int
    completed: int
    unavailable: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "destination": self.destination,
            "total": self.total,
            "completed": self.completed,
            "unavailable": self.unavailable,
        }


def export_predictions_jsonl(
    adapter: ModelAdapter,
    samples: Sequence[BatchPredictionSample],
    destination: str | Path,
    *,
    fail_fast: bool = False,
) -> PredictionExportSummary:
    """Run samples and atomically write evaluator-compatible JSONL.

    A failed sample remains visible as an explicit unavailable record. Its
    payload is deliberately invalid for scoring, so the evaluator counts it as
    schema-invalid instead of silently treating it as a model prediction.
    Cancellation-like ``BaseException`` values are not intercepted.
    """

    identifiers = [sample.sample_id for sample in samples]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample_id values must be unique")

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    completed = 0
    unavailable = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for sample in samples:
                try:
                    result = adapter.analyze(sample.request)
                    entry = {
                        "sample_id": sample.sample_id,
                        "source": "model_result",
                        "prediction": model_result_to_dict(result),
                        "status": "completed",
                    }
                    completed += 1
                except Exception as exc:
                    if fail_fast:
                        raise
                    entry = {
                        "sample_id": sample.sample_id,
                        "source": "model_result",
                        "prediction_raw": "unavailable",
                        "status": "unavailable",
                        "error": {
                            "type": type(exc).__name__,
                            "message": "prediction unavailable",
                        },
                    }
                    unavailable += 1
                stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return PredictionExportSummary(
        destination=str(target),
        total=len(samples),
        completed=completed,
        unavailable=unavailable,
    )
