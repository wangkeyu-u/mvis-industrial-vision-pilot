"""CLI for fair paired baseline/candidate comparisons."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from .cli import build_records, load_ground_truth, load_predictions
from .comparison import compare_model_reports
from .reporting import evaluate_offline_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare candidate and baseline fairly")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    parser.add_argument("--minimum-reliable-samples", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    truths = load_ground_truth(arguments.ground_truth)
    baseline = evaluate_offline_records(
        build_records(truths, load_predictions(arguments.baseline_predictions)),
        iou_threshold=arguments.iou_threshold,
    )
    candidate = evaluate_offline_records(
        build_records(truths, load_predictions(arguments.candidate_predictions)),
        iou_threshold=arguments.iou_threshold,
    )
    comparison = compare_model_reports(
        baseline,
        candidate,
        resamples=arguments.bootstrap_resamples,
        seed=arguments.bootstrap_seed,
        minimum_reliable_samples=arguments.minimum_reliable_samples,
    )
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(comparison.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    print(json.dumps({"conclusion_strength": comparison.conclusion_strength, "notice": comparison.notice}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
