from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from src.core.model_registry import AdapterRequest, MockModelAdapter
from src.core.schemas import AnalyzeOptions, AnalyzeTask, ImageMetadata, ModelOutput
from src.observability import real_validation

from .conftest import image_bytes


def blocked_diagnostics() -> dict:
    return {
        "ready": False,
        "status": "unavailable",
        "reasons": [
            "dependency_missing:mlx",
            "dependency_missing:mlx_vlm",
            "pinned_model_snapshot_unavailable",
        ],
        "environment": {
            "python": "3.13.13",
            "system": "Darwin",
            "release": "25.5.0",
            "machine": "arm64",
            "total_memory_bytes": 16 * 1024**3,
        },
        "dependencies": {
            "mlx": {"installed": False, "version": None},
            "mlx_vlm": {"installed": False, "version": None},
        },
        "model": {
            "alias": "qwen3-vl-2b-instruct-4bit",
            "model_id": "mlx-community/Qwen3-VL-2B-Instruct-4bit",
            "base_model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "9c4f5209e57b31f4b9dfba735de3fb983739c9cc",
            "backend": "mlx_vlm",
            "config_fingerprint": "71b87d18c01aeb4d",
            "allow_download": False,
            "trust_remote_code": False,
        },
        "quantization": {"bits": 4, "mode": "affine"},
        "cache": {
            "source": "huggingface_cache",
            "exists": False,
            "complete": False,
            "config_present": False,
            "weight_files": [],
            "reason": "snapshot_directory_missing",
            "path": "/private/path/must-not-leak",
        },
    }


def test_real_acceptance_blocks_before_registry_and_never_substitutes_mock(
    settings, tmp_path, monkeypatch
) -> None:
    sample = tmp_path / "probe.jpg"
    sample.write_bytes(image_bytes())
    monkeypatch.setattr(
        "src.inference.diagnostics.diagnose_model",
        lambda _path: blocked_diagnostics(),
    )

    def unexpected_registry(_settings):
        raise AssertionError("registry must not load when diagnostics is blocked")

    monkeypatch.setattr(real_validation, "build_service_registry", unexpected_registry)
    report = real_validation.run_real_runtime_acceptance(
        replace(settings, model_mode="real"),
        sample_path=sample,
        warmup_runs=5,
        measured_runs=30,
    )

    assert report["status"] == "blocked"
    assert report["runtime_qualification"] == {
        "requested_mode": "real",
        "real_ready": False,
        "mock_used": False,
        "auto_fallback_allowed": False,
    }
    assert report["performance"]["eligible"] is False
    assert report["performance"]["measured_runs"] == 0
    assert report["probes"]["measured"] == []
    assert "path" not in report["diagnostics"]["cache"]


def test_phase5_markdown_and_json_preserve_unqualified_blocked_evidence(
    settings, tmp_path, monkeypatch
) -> None:
    sample = tmp_path / "probe.jpg"
    sample.write_bytes(image_bytes())
    monkeypatch.setattr(
        "src.inference.diagnostics.diagnose_model",
        lambda _path: blocked_diagnostics(),
    )
    report = real_validation.run_real_runtime_acceptance(
        replace(settings, model_mode="real"), sample_path=sample
    )
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    real_validation.write_real_runtime_reports(
        report, json_path=json_path, markdown_path=markdown_path
    )

    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert reloaded["status"] == "blocked"
    assert "Mock used as real evidence: `false`" in markdown
    assert "Performance eligible: `false`" in markdown
    assert "9c4f5209e57b31f4b9dfba735de3fb983739c9cc" in markdown
    assert "dependency_missing:mlx" in markdown


def test_timeout_once_control_recovers_to_delegate() -> None:
    async def scenario() -> None:
        adapter = real_validation.TimeoutOnceAdapter(MockModelAdapter())
        request = AdapterRequest(
            image_bytes=image_bytes(),
            image=ImageMetadata(
                width=32,
                height=24,
                format="jpeg",
                mode="RGB",
                byte_size=len(image_bytes()),
            ),
            request_id="req_phase5",
            query="inspect",
            task=AnalyzeTask.INSPECT,
            use_specialist=False,
            options=AnalyzeOptions(),
        )
        adapter.arm()
        with pytest.raises(TimeoutError, match="controlled timeout"):
            await adapter.analyze(request)
        recovered = ModelOutput.model_validate(await adapter.analyze(request))
        assert recovered.result == "violation"

    asyncio.run(scenario())


def test_real_response_qualification_rejects_mock_identity_and_warning() -> None:
    assert real_validation._is_real_response(  # noqa: SLF001
        {
            "status_code": 200,
            "model": {"base": "qwen3-vl-2b-instruct-4bit"},
            "warnings": [],
        }
    )
    assert not real_validation._is_real_response(  # noqa: SLF001
        {
            "status_code": 200,
            "model": {"base": "mock-vlm-0"},
            "warnings": [],
        }
    )
    assert not real_validation._is_real_response(  # noqa: SLF001
        {
            "status_code": 200,
            "model": {"base": "qwen3-vl-2b-instruct-4bit"},
            "warnings": ["mock_adapter"],
        }
    )
