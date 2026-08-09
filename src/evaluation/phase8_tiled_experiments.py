"""Phase 8 tiled-heatmap experiment driver.

Flattens the tiled PatchCore scores (one fused native-resolution heatmap per
fusion mode per sample) and runs the validation-only post-processing grid per
fusion mode. The winner across modes is evaluated on test exactly once.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.localization_postprocess import PostprocessConfig
from src.evaluation.phase8_experiments import (
    experiment_grid,
    run_grid,
    select_best_config,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TILED = ROOT / "artifacts/model/phase8/patchcore_tiled"
DEFAULT_DATASET = ROOT / "data/processed/ksdd_v0"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase8/tiled_postprocess_experiments"


def _flatten_split(
    tiled_scores: Path, fusion_mode: str, split: str, workspace: Path
) -> Path:
    flattened = workspace / f"{fusion_mode}_{split}.jsonl"
    with tiled_scores.open("r", encoding="utf-8") as source, flattened.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["split"] != split:
                continue
            fusion = record["fusion"][fusion_mode]
            target.write(
                json.dumps(
                    {
                        "split": split,
                        "sample_id": record["sample_id"],
                        "anomaly_score": record["anomaly_score"],
                        "heatmap_npy": fusion["heatmap_npy"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return flattened


def run_tiled_experiments(
    tiled_dir: str | Path = DEFAULT_TILED,
    dataset_root: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    tiled = Path(tiled_dir).resolve()
    dataset = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite tiled experiments: {output}")
    output.mkdir(parents=True)

    tiled_manifest = json.loads((tiled / "run_manifest.json").read_text(encoding="utf-8"))
    image_threshold = float(tiled_manifest["calibration"]["image_threshold"])
    model_fingerprint = tiled_manifest["artifacts"]["checkpoint_sha256"]
    fusion_modes = tuple(tiled_manifest["configuration"]["fusion_modes"])
    grid = experiment_grid(
        min_area_values=(8, 64, 256),
        merge_options=((0, 0.0), (12, 3.0), (24, 3.0)),
    )

    validation_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="phase8_tiled_") as workspace_name:
        workspace = Path(workspace_name)
        for mode in fusion_modes:
            flattened = _flatten_split(tiled / "scores.jsonl", mode, "validation", workspace)
            results = run_grid(
                heatmap_source=flattened,
                dataset_root=dataset,
                split="validation",
                image_threshold=image_threshold,
                output_path=output / f"validation_grid_{mode}.jsonl",
                model_fingerprint=model_fingerprint,
                configs=grid,
            )
            for result in results:
                result["fusion_mode"] = mode
            validation_results.extend(results)

    best = select_best_config(validation_results)
    best_config = PostprocessConfig(**best["config"])
    best_mode = best["fusion_mode"]
    with tempfile.TemporaryDirectory(prefix="phase8_tiled_test_") as workspace_name:
        flattened_test = _flatten_split(
            tiled / "scores.jsonl", best_mode, "test", Path(workspace_name)
        )
        test_results = run_grid(
            heatmap_source=flattened_test,
            dataset_root=dataset,
            split="test",
            image_threshold=image_threshold,
            output_path=output / "test_selected.jsonl",
            model_fingerprint=model_fingerprint,
            configs=[best_config],
        )
    test_results[0]["fusion_mode"] = best_mode
    (output / "test_selected.jsonl").write_text(
        json.dumps(test_results[0], sort_keys=True) + "\n", encoding="utf-8"
    )

    selection = {
        "schema_version": "1.0",
        "phase": "phase8",
        "kind": "tiled_patchcore_selection",
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
        "candidates_evaluated": len(validation_results),
        "image_threshold": image_threshold,
        "selected_config": best["config"],
        "selected_config_fingerprint": best["config_fingerprint"],
        "selected_fusion_mode": best_mode,
        "validation_metrics": best["metrics"],
        "test_metrics": test_results[0]["metrics"],
        "tiled_source_manifest": str(tiled / "run_manifest.json"),
        "model_fingerprint": model_fingerprint,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    return selection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiled-dir", type=Path, default=DEFAULT_TILED)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    selection = run_tiled_experiments(args.tiled_dir, args.dataset_root, args.output_dir)
    print(json.dumps({
        "candidates": selection["candidates_evaluated"],
        "selected_fusion_mode": selection["selected_fusion_mode"],
        "selected_config": selection["selected_config"]["name"],
        "validation_acc_at_iou": selection["validation_metrics"]["acc_at_iou_0_5"],
        "test_acc_at_iou": selection["test_metrics"]["acc_at_iou_0_5"],
        "test_pixel_dice": selection["test_metrics"]["pixel_dice"],
        "test_box_f1": selection["test_metrics"]["box_f1"],
        "test_image_macro_f1": selection["test_metrics"]["image_macro_f1"],
        "test_hard_negative_fpr": selection["test_metrics"]["hard_negative_fpr"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
