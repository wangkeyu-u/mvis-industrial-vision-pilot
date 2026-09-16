# AI-assisted development

Codex assisted the September 2026 repository review, numerical test setup repair and organization of the existing model reports. Earlier commits and attribution remain available.

## Evidence from this revision

- PatchCore localization failures, U-Net results and corrected entity-grouped CV numbers come from the retained historical reports. The review did not rerun training or recreate absent weights.
- A clean checkout exposed missing numerical dependencies and a test output path that assumed an existing artifacts directory. The [repair record](failures/003-clean-checkout.md) describes the changes and validation.
- The [experiment index](experiments/baseline.md) separates those historical model results from the 278 passing software tests, 8 skips and 12 passing subtests.

The developer is responsible for accepting the evaluation protocol and any deployment decision. The documentation does not establish who authored each historical experiment, and `production_ready=false` remains in force.
