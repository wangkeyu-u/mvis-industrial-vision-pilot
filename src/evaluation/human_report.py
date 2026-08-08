"""Render accessible Markdown and standalone HTML evaluation reports."""

from __future__ import annotations

import html
from typing import Any, Mapping

from .reporting import EvaluationReport

METRICS = (
    "macro_f1",
    "acc_at_iou",
    "json_schema_validity_rate",
    "hard_negative_false_positive_rate",
    "evidence_conclusion_consistency_rate",
)


def _format(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}" if isinstance(value, float) else str(value)


def _watermark(fixture_only: bool, mock_only: bool) -> str | None:
    if fixture_only and mock_only:
        return "SYNTHETIC FIXTURE / MOCK OUTPUT — NOT MODEL PERFORMANCE"
    if fixture_only:
        return "SYNTHETIC FIXTURE — NOT MODEL PERFORMANCE"
    if mock_only:
        return "MOCK OUTPUT — NOT MODEL PERFORMANCE"
    return None


def render_markdown_report(
    report: EvaluationReport,
    metrics_payload: Mapping[str, Any],
    failures: Mapping[str, Any],
    *,
    comparison: Mapping[str, Any] | None = None,
    fixture_only: bool = False,
    mock_only: bool = False,
) -> str:
    watermark = _watermark(fixture_only, mock_only)
    lines = ["# Visual Compliance Evaluation Report", ""]
    if watermark:
        lines.extend([f"> **{watermark}**", ""])
    lines.extend(
        [
            "## Executive summary",
            "",
            f"- Samples: {report.summary['sample_count']}",
            f"- Ground-truth localization targets: {report.summary['localization_target_count']}",
            f"- Hard negatives: {report.summary['hard_negative_count']}",
            f"- Failure cases: {report.summary['failure_case_count']}",
            "",
            "## Core metrics and confidence intervals",
            "",
            "| Metric | Estimate | Lower | Upper | Observations |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    intervals = metrics_payload["confidence_intervals"]
    for metric in METRICS:
        interval = intervals[metric]
        lines.append(
            f"| {metric} | {_format(interval['point_estimate'])} | {_format(interval['lower'])} | {_format(interval['upper'])} | {interval['observation_count']} |"
        )
    lines.extend(["", "## KPI-01 to KPI-05 acceptance", "", "| KPI | Metric | Status | Reasons |", "|---|---|---|---|"])
    for item in metrics_payload["kpi_acceptance"]["assessments"]:
        reasons = "; ".join(reason["message"] for reason in item["reasons"]) or "All requirements met"
        lines.append(f"| {item['kpi_id']} | {item['metric']} | {item['status']} | {reasons} |")
    lines.extend(["", "## Slice differences", "", "| Slice | Samples | Metric | Value | Delta vs overall |", "|---|---:|---|---:|---:|"])
    for slice_name, slice_values in sorted(report.slice_metrics.items()):
        for metric in METRICS:
            value = float(slice_values[metric])
            delta = value - float(report.summary[metric])
            lines.append(f"| {slice_name} | {slice_values['sample_count']} | {metric} | {value:.4f} | {delta:+.4f} |")
    lines.extend(["", "## Failure cases", "", "| Sample | Failure codes | Source |", "|---|---|---|"])
    for sample in report.failure_cases:
        lines.append(
            f"| {sample.sample_id} | {', '.join(sample.failure_codes)} | {sample.prediction.source_kind} |"
        )
    if not report.failure_cases:
        lines.append("| _None_ |  |  |")
    lines.extend(["", "### Failure counts by slice", "", "| Slice | Samples | Failures | Rate |", "|---|---:|---:|---:|"])
    for slice_name, value in sorted(failures["by_slice"].items()):
        lines.append(f"| {slice_name} | {value['sample_count']} | {value['failure_count']} | {value['failure_rate']:.2%} |")
    if comparison is not None:
        lines.extend(["", "## Candidate model comparison", "", f"Comparison strength: **{comparison['conclusion_strength']}**", "", comparison["notice"], "", "| Metric | Baseline | Candidate | Delta | Hint |", "|---|---:|---:|---:|---|"])
        for metric, value in comparison["metrics"].items():
            lines.append(f"| {metric} | {value['baseline']:.4f} | {value['candidate']:.4f} | {value['delta']:+.4f} | {value['significance_hint']} |")
    lines.extend(["", "## Interpretation limits", "", "- Confidence intervals quantify sampling uncertainty, not dataset bias or annotation error.", "- Slice comparisons are descriptive and are not corrected for multiple comparisons.", "- Fixture or mock watermarked reports must never be presented as model performance.", ""])
    return "\n".join(lines)


def render_html_report(
    report: EvaluationReport,
    metrics_payload: Mapping[str, Any],
    failures: Mapping[str, Any],
    *,
    comparison: Mapping[str, Any] | None = None,
    fixture_only: bool = False,
    mock_only: bool = False,
) -> str:
    watermark = _watermark(fixture_only, mock_only)
    intervals = metrics_payload["confidence_intervals"]

    def rows(values: list[list[Any]]) -> str:
        return "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
            for row in values
        )

    metric_rows = [
        [metric, _format(intervals[metric]["point_estimate"]), _format(intervals[metric]["lower"]), _format(intervals[metric]["upper"]), intervals[metric]["observation_count"]]
        for metric in METRICS
    ]
    kpi_rows = [
        [item["kpi_id"], item["metric"], item["status"], "; ".join(reason["message"] for reason in item["reasons"]) or "All requirements met"]
        for item in metrics_payload["kpi_acceptance"]["assessments"]
    ]
    slice_rows = [
        [slice_name, values["sample_count"], metric, f"{float(values[metric]):.4f}", f"{float(values[metric]) - float(report.summary[metric]):+.4f}"]
        for slice_name, values in sorted(report.slice_metrics.items())
        for metric in METRICS
    ]
    failure_rows = [[sample.sample_id, ", ".join(sample.failure_codes), sample.prediction.source_kind] for sample in report.failure_cases] or [["None", "", ""]]
    comparison_section = ""
    if comparison is not None:
        comparison_rows = [[metric, f"{value['baseline']:.4f}", f"{value['candidate']:.4f}", f"{value['delta']:+.4f}", value["significance_hint"]] for metric, value in comparison["metrics"].items()]
        comparison_section = f"<h2>Candidate model comparison</h2><p><strong>Strength: {html.escape(comparison['conclusion_strength'])}</strong></p><p>{html.escape(comparison['notice'])}</p><table><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Hint</th></tr></thead><tbody>{rows(comparison_rows)}</tbody></table>"
    watermark_html = f'<div class="watermark" role="alert">{html.escape(watermark)}</div>' if watermark else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual Compliance Evaluation Report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a}}table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}th,td{{border:1px solid #ccd1d1;padding:.5rem;text-align:left}}th{{background:#f4f6f7}}.watermark{{background:#7b241c;color:white;font-weight:800;padding:1rem;border:4px solid #e6b0aa;margin-bottom:1rem}}code{{background:#f4f6f7;padding:.1rem .25rem}}</style></head><body>
{watermark_html}<h1>Visual Compliance Evaluation Report</h1>
<h2>Executive summary</h2><ul><li>Samples: {report.summary['sample_count']}</li><li>Localization targets: {report.summary['localization_target_count']}</li><li>Hard negatives: {report.summary['hard_negative_count']}</li><li>Failure cases: {report.summary['failure_case_count']}</li></ul>
<h2>Core metrics and confidence intervals</h2><table><thead><tr><th>Metric</th><th>Estimate</th><th>Lower</th><th>Upper</th><th>Observations</th></tr></thead><tbody>{rows(metric_rows)}</tbody></table>
<h2>KPI acceptance</h2><table><thead><tr><th>KPI</th><th>Metric</th><th>Status</th><th>Reasons</th></tr></thead><tbody>{rows(kpi_rows)}</tbody></table>
<h2>Slice differences</h2><table><thead><tr><th>Slice</th><th>Samples</th><th>Metric</th><th>Value</th><th>Delta</th></tr></thead><tbody>{rows(slice_rows)}</tbody></table>
<h2>Failure cases</h2><table><thead><tr><th>Sample</th><th>Failure codes</th><th>Source</th></tr></thead><tbody>{rows(failure_rows)}</tbody></table>
{comparison_section}<h2>Interpretation limits</h2><ul><li>Intervals do not capture dataset bias or annotation error.</li><li>Slice comparisons are descriptive and not multiplicity-corrected.</li><li>Fixture/mock reports are not model performance.</li></ul></body></html>"""
