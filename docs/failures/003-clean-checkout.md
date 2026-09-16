# Clean-checkout tests relied on undeclared dependencies and local artifacts

## Symptom
2026-09-15 clean `uv sync --group dev; uv run pytest -q` failed collection in five modules because NumPy was absent; after adding it, OpenCV and a missing artifacts directory caused failures.

## Reproduction
Run the documented install/test commands in a fresh clone, before this revision.

## Root Cause
Numerical test dependencies were absent from the dev group; a test wrote a temporary file into a local artifacts directory assumed to exist.

## Attempts
Preserved the initial failing output in the local audit; added numerical dev dependencies and replaced the test path with pytest tmp_path.

## Final Fix
Dev dependencies are now locked; the test uses an isolated temporary directory. Current results are in [validation](../experiments/validation.md).

## Remaining Risk
Model/data-dependent checks may still skip; passing contracts does not reproduce training metrics.
