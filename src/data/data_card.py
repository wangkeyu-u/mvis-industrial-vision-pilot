"""Generate a compact Markdown data card from verified dataset artifacts."""

from __future__ import annotations

from typing import Any, Mapping

from .manifest import DatasetManifest


def _table(counts: Mapping[str, Any]) -> str:
    rows = ["| Value | Count |", "|---|---:|"]
    rows.extend(f"| {name} | {count} |" for name, count in sorted(counts.items()))
    return "\n".join(rows) if len(rows) > 2 else "_No entries._"


def generate_data_card(
    manifest: DatasetManifest,
    statistics: Mapping[str, Any],
    *,
    title: str = "Visual Compliance Dataset",
    fixture_only: bool = False,
) -> str:
    total = int(statistics.get("total_samples", 0))
    hard_negative_rate = float(statistics.get("hard_negative_rate", 0.0))
    watermark = (
        "> **SYNTHETIC FIXTURE — NOT TRAINING OR MODEL EVALUATION DATA**\n\n"
        if fixture_only
        else ""
    )
    limitations = []
    if total < 300:
        limitations.append(
            "The dataset has fewer than 300 samples; quantitative conclusions require confidence intervals and reduced claim strength."
        )
    if hard_negative_rate < 0.20:
        limitations.append("Hard negatives are below the 20% recommended test-set coverage.")
    limitations.append("dHash near-duplicate detection may miss heavy crops, composites, or overlays.")
    return f"""# {title}

{watermark}## Identity

- Dataset version: `{manifest.dataset_version}`
- Schema version: `{manifest.schema_version}`
- Created at: `{manifest.created_at}`
- Frozen test set: `{str(manifest.frozen_test).lower()}`
- Total samples: {total}
- Hard negatives: {statistics.get('hard_negative_count', 0)} ({hard_negative_rate:.1%})

## Intended use

Single-image visual compliance classification, evidence grounding, structured-output validation, and offline evaluation. It must not be used for autonomous high-risk decisions.

## Splits

{_table(statistics.get('split_counts', {}))}

## Result distribution

{_table(statistics.get('result_counts', {}))}

## Difficulty and slice coverage

{_table(statistics.get('slice_counts', {}))}

## Sources and licenses

### Sources

{_table(statistics.get('source_counts', {}))}

### Licenses

{_table(statistics.get('license_counts', {}))}

All imported samples passed the configured license allow/deny policy. Unknown or denied licenses are excluded and recorded in the import rejection report.

## Known limitations

{chr(10).join(f'- {item}' for item in limitations)}

## Reproducibility

The accompanying manifest records entity-isolated splits, SHA-256 file hashes, 64-bit dHash values, source identifiers, and license identifiers. Any frozen-test change requires a new dataset version.
"""
