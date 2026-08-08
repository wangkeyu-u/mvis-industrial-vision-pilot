from __future__ import annotations

import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HTML = (PROJECT_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
APP = (PROJECT_ROOT / "ui" / "app.mjs").read_text(encoding="utf-8")
CSS = (PROJECT_ROOT / "ui" / "styles.css").read_text(encoding="utf-8")


class UIContractTest(unittest.TestCase):
    def test_required_controls_and_result_surfaces_exist(self) -> None:
        required_ids = {
            "image-input",
            "demo-sample",
            "query-input",
            "task-select",
            "model-select",
            "specialist-toggle",
            "overlay-canvas",
            "result-card",
            "evidence-list",
            "warning-list",
            "latency-total",
            "request-id",
            "download-json",
            "download-evidence",
            "runtime-status",
            "runtime-label",
            "toggle-evaluation",
            "evaluation-panel",
            "evaluation-input",
            "runtime-diagnostics",
            "evaluation-watermark",
            "evaluation-comparison",
            "evaluation-kpis",
            "evaluation-slices",
            "evaluation-failures",
        }
        found_ids = set(re.findall(r'id="([^"]+)"', HTML))
        self.assertEqual(required_ids - found_ids, set())

    def test_all_policy_states_are_present_in_ui_logic(self) -> None:
        for state in ("violation", "compliant", "uncertain", "refused", "error"):
            with self.subTest(state=state):
                self.assertIn(f"{state}:", APP)
                self.assertIn(f'[data-state="{state}"]', CSS)

    def test_accessibility_and_responsive_contract(self) -> None:
        self.assertIn('class="skip-link"', HTML)
        self.assertIn('role="alert"', HTML)
        self.assertIn('aria-live="polite"', HTML)
        self.assertIn("prefers-reduced-motion", CSS)
        self.assertGreaterEqual(CSS.count("@media (max-width:"), 2)

    def test_export_and_original_coordinate_overlay_are_implemented(self) -> None:
        self.assertIn("JSON.stringify(result, null, 2)", APP)
        self.assertIn("prepareJsonExport(result)", APP)
        self.assertIn('setAttribute("aria-disabled", "true")', APP)
        self.assertIn("imageDimensions.width", APP)
        self.assertIn("object.bbox", APP)
        self.assertIn("object.confidence", APP)
        self.assertIn("evidence-confidence", APP)
        self.assertIn("scaleX", APP)
        self.assertIn("scaleY", APP)

    def test_live_api_mode_exposes_transport_and_cancellation_controls(self) -> None:
        self.assertIn('params.get("transport")', APP)
        self.assertIn("requestTimeoutMs", APP)
        self.assertIn('event.key === "Escape"', APP)
        self.assertIn('firstElementChild.textContent = "取消请求"', APP)
        self.assertIn("policy_state", APP)

    def test_readiness_truth_states_and_acceptance_evidence_are_visible(self) -> None:
        for state in ("mock", "degraded", "real", "unavailable"):
            self.assertIn(f'[data-state="{state}"]', CSS)
        self.assertIn("refreshReadiness", APP)
        self.assertIn("mvis_demo_acceptance_evidence", APP)
        self.assertIn("stale_result_protection", APP)
        self.assertIn("image_bytes_embedded: false", APP)
        self.assertIn("query_text_embedded: false", APP)
        self.assertIn("resetAcceptanceExport", APP)
        self.assertIn("loadDemoSample", APP)

    def test_evaluation_portfolio_is_lazy_accessible_and_visibly_fail_closed(self) -> None:
        evaluation = (PROJECT_ROOT / "ui" / "evaluation-panel.mjs").read_text(encoding="utf-8")
        parser = (PROJECT_ROOT / "src" / "client" / "evaluation-report.mjs").read_text(encoding="utf-8")
        self.assertIn('aria-controls="evaluation-panel"', HTML)
        self.assertIn('aria-labelledby="evaluation-heading"', HTML)
        self.assertIn('role="alert"', HTML)
        self.assertIn("FIXTURE / MOCK", parser)
        self.assertIn("已清空上一份评测证据", evaluation)
        self.assertIn("PACKAGE_HASH_MISMATCH", parser)
        self.assertIn("eligibleForModelAcceptance", parser)
        self.assertIn('event.key === "Escape"', evaluation)
        self.assertIn("evaluation-panel", CSS)
        self.assertIn("evaluation-watermark", CSS)


if __name__ == "__main__":
    unittest.main()
