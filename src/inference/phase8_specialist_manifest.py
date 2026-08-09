"""Phase 8: assemble the bridge-consumable U-Net specialist manifest.

Combines the frozen U-Net run manifest and the validation-only selection
into one hash-pinned specialist manifest consumed by
``create_specialist_service_adapter``. Pure assembly: no metrics are
recomputed here, and selection evidence must come from the frozen artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.inference.patchcore_specialist import _sha256

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "artifacts/model/phase8/unet_resnet18"
DEFAULT_SELECTION = ROOT / "artifacts/model/phase8/unet_postprocess_experiments/selection.json"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase8/specialist_manifest_unet.json"


def build_phase8_manifest(
    run_dir: str | Path = DEFAULT_RUN,
    selection_path: str | Path = DEFAULT_SELECTION,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    run_manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if run_manifest.get("kind") != "unet_supervised_segmentation":
        raise ValueError("run manifest is not the phase 8 U-Net run")
    if selection.get("test_labels_used_for_selection") is not False:
        raise ValueError("selection evidence must not use test labels")
    checkpoint = Path(run_manifest["artifacts"]["checkpoint"])
    checkpoint_sha = run_manifest["artifacts"]["checkpoint_sha256"]
    if _sha256(checkpoint) != checkpoint_sha:
        raise ValueError("U-Net checkpoint hash mismatch against its run manifest")
    config = run_manifest["configuration"]
    manifest = {
        "schema_version": "1.0",
        "phase": "phase8",
        "kind": "phase8_specialist_manifest",
        "algorithm": "unet",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_labels_used_for_selection": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "dataset_manifest_sha256": run_manifest["dataset"]["manifest_sha256"],
        "tile_manifest_sha256": run_manifest["dataset"]["tile_manifest_sha256"],
        "image_threshold": run_manifest["calibration"]["image_threshold"],
        "fusion_mode": selection["selected_fusion_mode"],
        "postprocess_config": selection["selected_config"],
        "postprocess_config_fingerprint": selection["selected_config_fingerprint"],
        "input_size": config["input_size"],
        "tile_size": config["tile_size"],
        "tile_stride": config["tile_stride"],
        "device": config["device"],
        "encoder": {
            "repository": run_manifest["encoder"]["repository"],
            "revision": run_manifest["encoder"]["revision"],
            "sha256": run_manifest["encoder"]["sha256"],
            "bytes": config["encoder_bytes"],
        },
        "quality_status": "pilot_candidate",
        "production_ready": False,
        "selection_evidence": {
            "run_manifest": str(run / "run_manifest.json"),
            "selection": str(Path(selection_path).resolve()),
            "selection_split": selection["selection_split"],
            "validation_metrics": selection["validation_metrics"],
        },
    }
    output = Path(output_path).resolve()
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest = build_phase8_manifest(args.run_dir, args.selection, args.output)
    print(json.dumps({
        "status": manifest["status"],
        "quality_status": manifest["quality_status"],
        "checkpoint_sha256": manifest["checkpoint_sha256"][:16],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
