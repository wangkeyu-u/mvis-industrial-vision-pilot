# Audit validation — 2026-09-15

Environment: macOS arm64, CPython 3.13.13, dependencies in `uv.lock`.

```bash
uv sync --group dev
uv run pytest -q
```

Result: **278 passed, 8 skipped, 12 subtests passed**, 64.47 seconds, one Starlette deprecation warning. Numerical dev dependencies and a test's temporary-file path were repaired after a fresh clone failed. See [failure and repair](../failures/003-clean-checkout.md).

This run validates available service, protocol, evaluation and numerical contracts. It does not train U-Net, rerun KSDD inference, or reproduce the historical model-quality numbers. Skipped optional paths do not count as validated model integrations. The reports linked in this directory retain historical results and their resource prerequisites.
