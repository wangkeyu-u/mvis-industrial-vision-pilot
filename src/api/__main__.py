"""Unified preflight, validation, and service entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from typing import Any

from src.core.config import ServiceSettings
from src.core.preflight import run_preflight
from src.core.registry_factory import build_service_registry


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["serve"]
    parser = _parser()
    args = parser.parse_args(arguments)

    settings = ServiceSettings.load(args.config)
    if args.command == "lifecycle-demo":
        _print_json(_lifecycle_demo())
        return 0
    if args.command == "real-validate":
        from src.observability.real_validation import (
            run_real_runtime_acceptance,
            write_real_runtime_reports,
        )

        real_settings = replace(settings, model_mode="real")
        report = run_real_runtime_acceptance(
            real_settings,
            sample_path=args.sample,
            host=args.host,
            port=args.port,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        )
        write_real_runtime_reports(
            report,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        _print_json(report)
        return 0 if report["status"] == "passed" else 2
    if args.command == "candidate-probe":
        from src.observability.candidate_validation import run_lora_candidate_probe

        real_settings = replace(settings, model_mode="real")
        report = run_lora_candidate_probe(
            real_settings,
            sample_path=args.sample,
            run_manifest_path=args.run_manifest,
        )
        _print_json(report)
        return 0 if report["status"] == "passed" else 2

    registry = build_service_registry(settings)
    preflight = run_preflight(settings, registry)

    if args.command == "preflight":
        _print_json(preflight)
        return 0 if preflight["can_serve"] else 2

    if args.command == "validate":
        from src.observability.validation import write_json_report

        report: dict[str, Any] = {
            "report_type": "mvis_stage4_validation",
            "report_version": "1.0.0",
            "preflight": preflight,
            "stability": None,
        }
        if preflight["can_serve"]:
            previous_suppression = os.environ.get("MVIS_SUPPRESS_CONFIG_LOG")
            os.environ["MVIS_SUPPRESS_CONFIG_LOG"] = "1"
            try:
                from src.api.app import create_app
            finally:
                if previous_suppression is None:
                    os.environ.pop("MVIS_SUPPRESS_CONFIG_LOG", None)
                else:
                    os.environ["MVIS_SUPPRESS_CONFIG_LOG"] = previous_suppression
            from src.observability.validation import sample_api_stability

            quiet_settings = replace(settings, log_level="WARNING")
            app = create_app(quiet_settings, registry, emit_config_log=False)
            report["stability"] = asyncio.run(
                sample_api_stability(app, quiet_settings, registry, request_count=args.requests)
            )
        if args.output:
            write_json_report(args.output, report)
        _print_json(report)
        stability = report["stability"] or {}
        passed = bool(
            preflight["can_serve"]
            and stability.get("stability", {}).get("success_count") == args.requests
        )
        return 0 if passed else 2

    if not preflight["can_serve"] and not args.allow_not_ready:
        _print_json(preflight, stream=sys.stderr)
        return 2

    import uvicorn

    from src.api.app import create_app

    uvicorn.run(
        create_app(settings, registry),
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVIS service lifecycle entry point")
    parser.add_argument(
        "--config", default=None, help="versioned service YAML (or MVIS_SERVICE_CONFIG)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="print machine-readable startup checks")
    subparsers.add_parser(
        "lifecycle-demo",
        help="run candidate/validate/activate/rollback using mock registrations",
    )
    real_validate = subparsers.add_parser(
        "real-validate",
        help="strict real-only Uvicorn acceptance; never falls back to mock",
    )
    real_validate.add_argument("--sample", default="assets/system_architecture.png")
    real_validate.add_argument("--host", default="127.0.0.1")
    real_validate.add_argument("--port", type=int, default=18086)
    real_validate.add_argument("--warmup-runs", type=int, default=5)
    real_validate.add_argument("--measured-runs", type=int, default=30)
    real_validate.add_argument(
        "--json-output",
        default="docs/deployment/phase5_real_runtime_report.json",
    )
    real_validate.add_argument(
        "--markdown-output",
        default="docs/deployment/phase5_real_runtime_report.md",
    )
    candidate_probe = subparsers.add_parser(
        "candidate-probe",
        help="load only the configured local LoRA candidate; never downloads or uses mock",
    )
    candidate_probe.add_argument(
        "--sample",
        default=None,
        help="optional JPEG/PNG/WebP used for one real pilot analyze probe",
    )
    candidate_probe.add_argument(
        "--run-manifest",
        default=None,
        help="optional algorithm training manifest included as bounded blocker evidence",
    )

    validate = subparsers.add_parser(
        "validate", help="run preflight and in-process API stability sampling"
    )
    validate.add_argument("--requests", type=int, default=100)
    validate.add_argument("--output", help="atomically write the JSON report")

    serve = subparsers.add_parser("serve", help="preflight then launch Uvicorn")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)
    serve.add_argument("--log-level", default="info")
    serve.add_argument(
        "--allow-not-ready",
        action="store_true",
        help="diagnostic only: start even when required preflight checks fail",
    )
    return parser


def _lifecycle_demo() -> dict[str, Any]:
    from src.core.model_registry import (
        MockModelAdapter,
        ModelRegistration,
        ModelState,
        build_mock_registry,
        fingerprint_mapping,
    )

    registry = build_mock_registry()
    candidate = ModelRegistration(
        model_id="mock-compliance-v1",
        adapter=MockModelAdapter(
            base="mock-vlm-1",
            adapter_id="mock-compliance-v1",
        ),
        state=ModelState.CANDIDATE,
        source="built-in-lifecycle-demo",
        quantization="none",
        config_fingerprint=fingerprint_mapping(
            {"demo": "modelops", "version": 1, "algorithm": "none"}
        ),
    )
    registry.register(candidate)
    registry.validate_candidate(
        candidate.model_id,
        actor="cli-demo",
        request_id="demo_validate",
        reason="dependency-free validation gate demonstration",
    )
    active_before = registry.active()
    registry.activate(
        candidate.model_id,
        actor="cli-demo",
        request_id="demo_activate",
        reason="atomic activation demonstration",
        expected_active_model_id=active_before.model_id if active_before else None,
        expected_active_fingerprint=(active_before.model_fingerprint if active_before else None),
    )
    after_activation = registry.statuses()
    registry.rollback(
        actor="cli-demo",
        request_id="demo_rollback",
        reason="atomic previous-alias rollback demonstration",
        expected_active_model_id=candidate.model_id,
        expected_active_fingerprint=candidate.model_fingerprint,
    )
    return {
        "report_type": "mvis_modelops_lifecycle_demo",
        "report_version": "1.0.0",
        "uses_real_weights": False,
        "after_activation": after_activation,
        "after_rollback": registry.statuses(),
        "audit": registry.audit_records(limit=100),
    }


def _print_json(report: dict[str, Any], *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    print(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        file=destination,
    )


if __name__ == "__main__":
    raise SystemExit(main())
