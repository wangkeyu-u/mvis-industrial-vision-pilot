"""Contract tests for phase 8 tiled PatchCore and U-Net configuration logic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.inference.patchcore_tiled import (
    _fuse_tile_heatmaps,
    load_tiled_config,
    tile_offsets,
)
from src.training.unet_segmentation import _grid_offsets, load_unet_config

ROOT = Path(__file__).resolve().parents[2]


def test_tiled_config_loads_and_validates() -> None:
    config = load_tiled_config(ROOT / "configs/models/ksdd_patchcore_resnet18_tiled_phase8.json")
    assert config.tile_size == 500
    assert config.fusion_modes == ("max", "mean", "weighted")
    payload = json.loads(
        (ROOT / "configs/models/ksdd_patchcore_resnet18_tiled_phase8.json").read_text()
    )
    payload["fusion_modes"] = ["median"]
    bad_path = ROOT / "artifacts" / "tmp_bad_tiled_config.json"
    bad_path.write_text(json.dumps(payload))
    try:
        with pytest.raises(ValueError):
            load_tiled_config(bad_path)
    finally:
        bad_path.unlink()


def test_unet_config_loads() -> None:
    config = load_unet_config(ROOT / "configs/models/ksdd_unet_resnet18_phase8.json")
    assert config.input_size == 256
    assert config.expected_train_tiles == 324


def test_tile_offsets_cover_image() -> None:
    assert tile_offsets(1265, 500) == (0, 382, 765)
    assert tile_offsets(500, 500) == (0,)
    assert tile_offsets(400, 500) == (0,)
    offsets = tile_offsets(1265, 500)
    assert offsets[-1] + 500 == 1265


def test_fusion_modes_preserve_shape_and_bounds() -> None:
    rng = np.random.default_rng(0)
    tile_maps = [rng.random((16, 16), dtype=np.float32) for _ in range(3)]
    offsets = (0, 20, 40)
    for mode in ("max", "mean", "weighted"):
        fused = _fuse_tile_heatmaps(tile_maps, offsets, (50, 90), 50, mode)
        assert fused.shape == (90, 50)
        assert float(fused.min()) >= 0.0
        assert float(fused.max()) <= 1.0 + 1e-6


def test_fusion_max_dominates_mean() -> None:
    tile_maps = [np.full((8, 8), 0.2, dtype=np.float32), np.full((8, 8), 0.9, dtype=np.float32)]
    offsets = (0, 4)
    fused_max = _fuse_tile_heatmaps(tile_maps, offsets, (20, 12), 8, "max")
    fused_mean = _fuse_tile_heatmaps(tile_maps, offsets, (20, 12), 8, "mean")
    overlap = slice(4, 8)
    assert np.all(fused_max[overlap] >= fused_mean[overlap] - 1e-6)
    assert float(fused_max[overlap].max()) == pytest.approx(0.9, abs=1e-3)


def test_grid_offsets_dense_cover() -> None:
    offsets = _grid_offsets(1265, 256, 128)
    assert offsets[0] == 0
    assert offsets[-1] == 1265 - 256
    for left, right in zip(offsets, offsets[1:]):
        assert right - left <= 128
    assert _grid_offsets(200, 256, 128) == (0,)
