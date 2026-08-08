import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  EvaluationImportError,
  parseEvaluationFiles,
  parseEvaluationPayload,
} from "../../src/client/evaluation-report.mjs";

const fixturePath = new URL("../../ui/fixtures/evaluation-package.fixture.json", import.meta.url);

async function fixtureBundle() {
  return JSON.parse(await readFile(fixturePath, "utf8"));
}

function jsonFile(name, value) {
  const text = JSON.stringify(value);
  const blob = new Blob([text], { type: "application/json" });
  Object.defineProperty(blob, "name", { value: name });
  return { file: blob, text };
}

test("fixture package is permanently watermarked and never acceptance eligible", async () => {
  const portfolio = parseEvaluationPayload(await fixtureBundle(), { filename: "fixture-package.json" });
  assert.equal(portfolio.truthState, "fixture");
  assert.match(portfolio.watermark, /FIXTURE \/ MOCK/);
  assert.equal(portfolio.eligibleForModelAcceptance, false);
  assert.equal(portfolio.acceptanceStatus, "fixture_only");
  assert.equal(portfolio.comparison.length, 5);
  assert.equal(portfolio.comparison[0].baseline, 0.7777777778);
  assert.deepEqual(portfolio.comparison[0].interval.lower, 0.44);
  assert.equal(portfolio.comparison[0].significanceHint, "insufficient_sample");
  assert.equal(portfolio.comparisonStrength, "degraded_small_sample");
  assert.equal(portfolio.slices[0].failureRate, 0.5);
  assert.equal(portfolio.failureCases.length, 2);
});

test("standalone evaluator report is parsed but marked unverified", async () => {
  const bundle = await fixtureBundle();
  const portfolio = parseEvaluationPayload(bundle.report, { filename: "report.json" });
  assert.equal(portfolio.kind, "report");
  assert.equal(portfolio.truthState, "unverified");
  assert.match(portfolio.watermark, /UNVERIFIED REPORT/);
  assert.equal(portfolio.eligibleForModelAcceptance, false);
  assert.equal(portfolio.comparison.every((item) => item.baseline === null), true);
  assert.equal(portfolio.failureCases[0].sampleId, "s-invalid-json");
});

test("mock-only package cannot escape the model-performance watermark", async () => {
  const bundle = await fixtureBundle();
  bundle.package_manifest.fixture_only = false;
  bundle.package_manifest.mock_only = true;
  bundle.metrics.fixture_only = false;
  bundle.metrics.mock_only = true;
  bundle.config.fixture_only = false;
  bundle.config.mock_only = true;
  bundle.metrics.kpi_acceptance.status = "mock_only";
  bundle.metrics.kpi_acceptance.eligible_for_model_acceptance = false;
  const portfolio = parseEvaluationPayload(bundle, { filename: "mock-package.json", verified: true });
  assert.equal(portfolio.truthState, "mock");
  assert.match(portfolio.watermark, /MOCK OUTPUT/);
  assert.equal(portfolio.eligibleForModelAcceptance, false);
});

test("complete package components verify hashes before allowing model acceptance", async () => {
  const bundle = await fixtureBundle();
  const metrics = structuredClone(bundle.metrics);
  metrics.fixture_only = false;
  metrics.fixture_notice = null;
  metrics.kpi_acceptance.status = "passed";
  metrics.kpi_acceptance.eligible_for_model_acceptance = true;
  metrics.kpi_acceptance.assessments.forEach((item) => { item.status = "passed"; item.reasons = []; });
  const metricsDocument = jsonFile("metrics.json", metrics);
  const reportDocument = jsonFile("report.json", bundle.report);
  const hashes = {
    "metrics.json": createHash("sha256").update(metricsDocument.text).digest("hex"),
    "report.json": createHash("sha256").update(reportDocument.text).digest("hex"),
    "report.md": "0".repeat(64),
  };
  const manifestDocument = jsonFile("package_manifest.json", {
    schema_version: "1.0.0",
    created_at: "2026-08-08T00:00:00Z",
    fixture_only: false,
    mock_only: false,
    fixture_notice: null,
    component_sha256: hashes,
  });
  const portfolio = await parseEvaluationFiles([
    manifestDocument.file,
    metricsDocument.file,
    reportDocument.file,
  ]);
  assert.equal(portfolio.verified, true);
  assert.equal(portfolio.truthState, "verified");
  assert.equal(portfolio.watermark, null);
  assert.equal(portfolio.eligibleForModelAcceptance, true);

  const tampered = jsonFile("report.json", { ...bundle.report, schema_version: "1.0.0", samples: [{ bad: true }] });
  await assert.rejects(
    parseEvaluationFiles([manifestDocument.file, metricsDocument.file, tampered.file]),
    (error) => error instanceof EvaluationImportError && error.code === "PACKAGE_HASH_MISMATCH",
  );
});

test("malformed metrics, unsupported schemas and unsafe file sets fail closed", async () => {
  const bundle = await fixtureBundle();
  assert.throws(
    () => parseEvaluationPayload({ ...bundle.report, schema_version: "2.0.0" }),
    (error) => error.code === "UNSUPPORTED_SCHEMA",
  );
  const invalid = structuredClone(bundle);
  invalid.metrics.summary.macro_f1 = 1.2;
  assert.throws(
    () => parseEvaluationPayload(invalid),
    (error) => error.code === "INVALID_METRIC",
  );
  const invalidComparison = structuredClone(bundle);
  invalidComparison.comparison.metrics.macro_f1.improvement_ci_lower = -1.2;
  assert.throws(
    () => parseEvaluationPayload(invalidComparison),
    (error) => error.code === "INVALID_METRIC",
  );
  await assert.rejects(
    parseEvaluationFiles([]),
    (error) => error.code === "NO_FILES",
  );
});
