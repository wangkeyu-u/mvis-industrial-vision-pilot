import assert from "node:assert/strict";
import test from "node:test";

import {
  RealAcceptanceError,
  buildRealAcceptanceArtifact,
  evaluateRealAcceptanceGate,
  requireRealAcceptance,
} from "../../src/client/real-acceptance.mjs";

function realReadiness() {
  const activeModel = {
    model_id: "qwen3-vl-active",
    base: "Qwen/Qwen3-VL-2B-Instruct",
    adapter: "compliance-lora-v1",
    revision: "9c4f5209",
    state: "active",
    ready: true,
    source: "model-config:qwen3-vl.json",
    quantization: "4-bit",
    config_fingerprint: "a".repeat(64),
    model_fingerprint: "b".repeat(64),
  };
  return {
    ready: true,
    state: "real",
    requestId: "req_ready_real",
    runtime: { requested_mode: "real", selected_mode: "real", degraded: false, fallback_reason: null },
    activeModel,
    aliases: { active: activeModel.model_id },
    models: [activeModel],
  };
}

test("real acceptance unlocks only for exact real non-degraded readiness", () => {
  const gate = evaluateRealAcceptanceGate(realReadiness(), { clientMode: "api" });
  assert.equal(gate.allowed, true);
  assert.equal(gate.label, "REAL MODEL EVIDENCE");
  assert.equal(gate.readinessRequestId, "req_ready_real");
  assert.equal(gate.model.revision, "9c4f5209");
});

test("degraded, non-real and offline clients remain screenshot blocked", () => {
  const degraded = realReadiness();
  degraded.runtime.degraded = true;
  degraded.runtime.fallback_reason = "model_load_failed";
  assert.deepEqual(
    evaluateRealAcceptanceGate(degraded).reasons.map((reason) => reason.code),
    ["RUNTIME_DEGRADED"],
  );
  const adapter = realReadiness();
  adapter.runtime.selected_mode = "model-test";
  assert.equal(evaluateRealAcceptanceGate(adapter).reasons.some((reason) => reason.code === "SELECTED_MODE_NOT_REAL"), true);
  assert.equal(evaluateRealAcceptanceGate(realReadiness(), { clientMode: "mock" }).allowed, false);
});

test("test-double source cannot obtain the real model seal", () => {
  const readiness = realReadiness();
  readiness.activeModel.source = "built-in-test-double";
  const gate = evaluateRealAcceptanceGate(readiness);
  assert.equal(gate.allowed, false);
  assert.equal(gate.reasons.some((reason) => reason.code === "TEST_DOUBLE_DETECTED"), true);
});

test("real artifact binds request, revision and latency without embedding input", () => {
  const artifact = buildRealAcceptanceArtifact({
    readiness: realReadiness(),
    clientMode: "api",
    result: {
      request_id: "req_analysis_real",
      result: "violation",
      objects: [{ bbox: [1, 2, 3, 4] }],
      uncertain: false,
      latency_ms: 941,
      timing: { preprocess_ms: 20, inference_ms: 900, validation_ms: 21 },
      model: { base: "Qwen/Qwen3-VL-2B-Instruct", revision: "result-revision" },
    },
    evaluation: { sourceName: "evaluation-package", truthState: "verified", verified: true, eligibleForModelAcceptance: true },
    input: {
      sampleId: "licensed-probe-1",
      licenseId: "cc-by-nc-sa-4.0",
      attribution: "dataset authors",
      sha256: "a".repeat(64),
    },
  });
  assert.equal(artifact.status, "observed");
  assert.equal(artifact.screenshot_eligible, true);
  assert.equal(artifact.model_performance_claim_allowed, true);
  assert.equal(artifact.observation.request_id, "req_analysis_real");
  assert.equal(artifact.observation.model_revision, "result-revision");
  assert.equal(artifact.observation.latency_ms, 941);
  assert.equal(artifact.observation.image_bytes_embedded, false);
  assert.equal(artifact.observation.query_text_embedded, false);
  assert.equal(artifact.input.sample_id, "licensed-probe-1");
  assert.equal(artifact.input.license_id, "cc-by-nc-sa-4.0");
  assert.equal(artifact.input.sha256, "a".repeat(64));
  assert.equal(artifact.input.image_bytes_embedded, false);
});

test("blocked artifact and assertion never fabricate a real observation", () => {
  const readiness = realReadiness();
  readiness.runtime.selected_mode = "mock-adapter";
  readiness.runtime.degraded = true;
  readiness.activeModel.source = "built-in-test-double";
  const artifact = buildRealAcceptanceArtifact({ readiness, clientMode: "api" });
  assert.equal(artifact.status, "blocked");
  assert.equal(artifact.screenshot_eligible, false);
  assert.equal(artifact.observation, null);
  assert.throws(
    () => requireRealAcceptance(readiness, { clientMode: "api" }),
    (error) => error instanceof RealAcceptanceError && error.code === "REAL_ACCEPTANCE_BLOCKED",
  );
});
