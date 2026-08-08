import assert from "node:assert/strict";
import test from "node:test";

import {
  AnalysisClientError,
  HttpAnalysisClient,
  MockAnalysisClient,
  classifyReadiness,
  createAnalysisClient,
  normalizeAnalysisResponse,
  validateImageFile,
} from "../../src/client/api-client.mjs";

function imageFile(type = "image/png", bytes = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3, 4]) {
  const blob = new Blob([Uint8Array.from(bytes)], { type });
  Object.defineProperty(blob, "name", { value: "case.png" });
  return blob;
}

function request(query) {
  return {
    image: imageFile(),
    imageWidth: 1000,
    imageHeight: 600,
    query,
    task: "inspect",
    model: "candidate",
    useSpecialist: true,
  };
}

test("client factory keeps mock and API implementations replaceable", () => {
  assert.equal(createAnalysisClient({ mode: "mock", latencyMs: 0 }).kind, "mock");
  const api = createAnalysisClient({ mode: "api", transport: "multipart", fetchImpl: async () => {} });
  assert.equal(api.kind, "api");
  assert.equal(api.transport, "multipart");
});

test("mock result exposes model, timing, warning, trace and bounded pixel box", async () => {
  const result = await new MockAnalysisClient({ latencyMs: 0 }).analyze(request("找出违规区域"));
  assert.equal(result.result, "violation");
  assert.deepEqual(result.objects[0].bbox, [560, 108, 820, 432]);
  assert.equal(result.objects[0].source, "fusion");
  assert.equal(result.model.alias, "candidate");
  assert.equal(result.timing.preprocess_ms + result.timing.inference_ms + result.timing.validation_ms, result.latency_ms);
  assert.ok(result.warnings.some((warning) => warning.includes("Mock")));
  assert.equal(result.trace.image_persisted, false);
});

test("mock exposes compliant, uncertain and refusal states without invented evidence", async () => {
  const client = new MockAnalysisClient({ latencyMs: 0 });
  for (const [query, expected] of [
    ["这是合规样例，请确认未发现违规", "compliant"],
    ["画面模糊且证据不足", "uncertain"],
    ["对身份证进行高风险自动判定", "refused"],
  ]) {
    const result = await client.analyze(request(query));
    assert.equal(result.result, expected);
    assert.deepEqual(result.objects, []);
  }
});

test("mock maps timeout demo to stable error code", async () => {
  const client = new MockAnalysisClient({ latencyMs: 0 });
  await assert.rejects(
    client.analyze(request("错误演示：模拟超时")),
    (error) => error instanceof AnalysisClientError && error.code === "INFERENCE_TIMEOUT" && error.status === 504,
  );
});

test("file validation rejects MIME/signature mismatch", async () => {
  await assert.rejects(
    validateImageFile(imageFile("image/png", [1, 2, 3, 4])),
    (error) => error.code === "INVALID_IMAGE" && error.message.includes("签名"),
  );
});

test("HTTP client sends base64 contract and preserves structured errors", async () => {
  let sentBody;
  const client = new HttpAnalysisClient({
    endpoint: "/v1/analyze",
    fetchImpl: async function (_url, options) {
      assert.equal(this, globalThis);
      sentBody = JSON.parse(options.body);
      return {
        ok: false,
        status: 422,
        json: async () => ({ request_id: "req_test", error: { code: "OUTPUT_VALIDATION_FAILED", message: "结构校验失败" } }),
      };
    },
  });
  await assert.rejects(
    client.analyze(request("找出违规区域")),
    (error) => error.code === "OUTPUT_VALIDATION_FAILED" && error.requestId === "req_test",
  );
  assert.ok(sentBody.image.startsWith("data:image/png;base64,"));
  assert.equal(sentBody.image_width, 1000);
  assert.equal(sentBody.query, "找出违规区域");
  assert.equal(sentBody.use_specialist, true);
});

test("HTTP multipart transport matches FastAPI form contract and forwards a safe request id", async () => {
  let sentOptions;
  const client = new HttpAnalysisClient({
    endpoint: "/v1/analyze",
    transport: "multipart",
    fetchImpl: async function (_url, options) {
      sentOptions = options;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          request_id: options.headers["X-Request-ID"],
          model: { base: "mock-vlm-0", adapter: "mock-compliance-v0" },
          result: "violation",
          objects: [{ label: "target", bbox: [1, 1, 8, 8], confidence: 0.5, source: "mock" }],
          reason: "FastAPI multipart response",
          uncertain: true,
          timing: { preprocess_ms: 1, inference_ms: 2, validation_ms: 1 },
          latency_ms: 4,
          warnings: ["mock_adapter"],
        }),
      };
    },
  });
  const result = await client.analyze(request("multipart inspection"));
  assert.ok(sentOptions.body instanceof FormData);
  assert.equal(sentOptions.headers["Content-Type"], undefined);
  assert.match(sentOptions.headers["X-Request-ID"], /^req_ui_[A-Za-z0-9_]+$/);
  assert.equal(sentOptions.body.get("query"), "multipart inspection");
  assert.equal(sentOptions.body.get("task"), "inspect");
  assert.equal(sentOptions.body.get("model"), "candidate");
  assert.equal(sentOptions.body.get("use_specialist"), "true");
  assert.deepEqual(JSON.parse(sentOptions.body.get("options")), { temperature: 0, seed: 42, max_tokens: 256 });
  assert.equal(sentOptions.body.get("image").type, "image/png");
  assert.equal(result.request_id, sentOptions.headers["X-Request-ID"]);
});

test("HTTP client distinguishes caller cancellation, timeout, and network unavailability", async () => {
  const abortingFetch = async function (_url, options) {
    return await new Promise((_, reject) => {
      if (options.signal.aborted) {
        reject(options.signal.reason);
        return;
      }
      options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true });
    });
  };
  const timeoutClient = new HttpAnalysisClient({ requestTimeoutMs: 5, fetchImpl: abortingFetch });
  await assert.rejects(
    timeoutClient.analyze(request("slow request")),
    (error) => error.code === "INFERENCE_TIMEOUT" && error.status === 504 && error.requestId.startsWith("req_ui_"),
  );

  const caller = new AbortController();
  const cancelClient = new HttpAnalysisClient({ requestTimeoutMs: 1000, fetchImpl: abortingFetch });
  const pending = cancelClient.analyze(request("cancel request"), { signal: caller.signal });
  caller.abort();
  await assert.rejects(pending, (error) => error.name === "AbortError");

  const offlineClient = new HttpAnalysisClient({ fetchImpl: async function () { throw new TypeError("network down"); } });
  await assert.rejects(
    offlineClient.analyze(request("offline request")),
    (error) => error.code === "MODEL_NOT_READY" && error.status === 503 && error.requestId.startsWith("req_ui_"),
  );
});

test("HTTP client accepts the real API response shape and derives missing total latency", async () => {
  const client = new HttpAnalysisClient({
    endpoint: "/v1/analyze",
    fetchImpl: async function () {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          schema_version: "1.0.0",
          request_id: "req_real_contract",
          model: { base: "mock-vlm-0", adapter: "mock-compliance-v0" },
          result: "violation",
          objects: [{ label: "blocked_exit", bbox: [8, 6, 24, 18], confidence: 0.88, source: "mock" }],
          reason: "detected evidence",
          uncertain: true,
          latency: { preprocess_ms: 2, inference_ms: 3, validation_ms: 4 },
          warnings: [],
        }),
      };
    },
  });
  const result = await client.analyze(request("找出不合规区域"));
  assert.equal(result.request_id, "req_real_contract");
  assert.equal(result.latency_ms, 9);
  assert.deepEqual(result.timing, result.latency);
  assert.equal(result.objects[0].confidence, 0.88);
});

test("success responses missing required fields fail closed", () => {
  assert.throws(
    () => normalizeAnalysisResponse({ request_id: "req_bad", result: "violation" }),
    (error) => error.code === "OUTPUT_VALIDATION_FAILED" && error.requestId === "req_bad",
  );
  assert.throws(
    () => normalizeAnalysisResponse({
      request_id: "req_bad_box",
      model: { base: "model" },
      result: "violation",
      objects: [{ label: "bad", bbox: [9, 1, 2, 8], confidence: 1.2, source: "mock" }],
      reason: "invalid coordinates",
    }),
    (error) => error.code === "OUTPUT_VALIDATION_FAILED" && error.requestId === "req_bad_box",
  );
});

test("machine-readable refusal warning maps uncertain API output to refused UI policy state", () => {
  const result = normalizeAnalysisResponse({
    request_id: "req_refusal",
    model: { base: "qwen-test", adapter: null },
    result: "uncertain",
    objects: [],
    reason: "insufficient evidence",
    uncertain: true,
    timing: { preprocess_ms: 1, inference_ms: 2, validation_ms: 1 },
    warnings: ["refusal:insufficient_evidence"],
  });
  assert.equal(result.result, "uncertain");
  assert.equal(result.policy_state, "refused");
});

test("readiness classification distinguishes real, degraded and unavailable truth states", () => {
  const base = {
    status: "ready",
    request_id: "req_ready",
    details: {
      aliases: { active: "model-v1" },
      models: [{ model_id: "model-v1", base: "qwen-real", source: "local-weights", ready: true }],
      runtime: { selected_mode: "real", degraded: false },
    },
  };
  assert.equal(classifyReadiness(base).state, "real");
  assert.equal(classifyReadiness(base).ready, true);
  const degraded = structuredClone(base);
  degraded.details.runtime = {
    selected_mode: "mock-adapter",
    degraded: true,
    fallback_reason: "interview demo",
  };
  degraded.details.models[0].source = "built-in-test-double";
  assert.equal(classifyReadiness(degraded).state, "degraded");
  assert.equal(classifyReadiness(degraded).detail, "interview demo");
  const nonAcceptance = structuredClone(base);
  nonAcceptance.details.runtime.selected_mode = "model-test";
  assert.equal(classifyReadiness(nonAcceptance).state, "degraded");
  assert.equal(classifyReadiness(nonAcceptance).label, "FastAPI 就绪 · 非验收模式");
  const unavailable = structuredClone(base);
  unavailable.status = "not_ready";
  unavailable.details.models[0].ready = false;
  assert.equal(classifyReadiness(unavailable).state, "unavailable");
  assert.equal(classifyReadiness(unavailable).ready, false);
  assert.equal(classifyReadiness({ status: "ready", backend: "mock", model_ready: false }).state, "unavailable");
});

test("HTTP readiness preflight preserves request id and degraded runtime evidence", async () => {
  let sent;
  const client = new HttpAnalysisClient({
    transport: "multipart",
    fetchImpl: async function (url, options) {
      sent = { url, options };
      return {
        ok: true,
        status: 200,
        headers: { get: () => options.headers["X-Request-ID"] },
        json: async () => ({
          status: "ready",
          request_id: options.headers["X-Request-ID"],
          details: {
            aliases: { active: "demo-v1" },
            models: [{ model_id: "demo-v1", base: "demo", source: "built-in-test-double", ready: true }],
            runtime: { selected_mode: "mock-adapter", degraded: true, fallback_reason: "demo only" },
          },
        }),
      };
    },
  });
  const readiness = await client.checkReadiness();
  assert.equal(sent.url, "/health/ready");
  assert.equal(sent.options.method, "GET");
  assert.match(sent.options.headers["X-Request-ID"], /^req_ui_/);
  assert.equal(readiness.state, "degraded");
  assert.equal(readiness.requestId, sent.options.headers["X-Request-ID"]);
});

test("HTTP readiness fails closed for 503 and network timeout", async () => {
  const notReady = new HttpAnalysisClient({
    fetchImpl: async function (_url, options) {
      return {
        ok: false,
        status: 503,
        headers: { get: () => options.headers["X-Request-ID"] },
        json: async () => ({
          status: "not_ready",
          request_id: options.headers["X-Request-ID"],
          details: { runtime: { selected_mode: "real" }, aliases: {}, models: [] },
        }),
      };
    },
  });
  await assert.rejects(notReady.checkReadiness(), (error) =>
    error.code === "MODEL_NOT_READY" && error.status === 503 && error.requestId.startsWith("req_ui_"));

  const hanging = new HttpAnalysisClient({
    requestTimeoutMs: 5,
    fetchImpl: async function (_url, options) {
      return await new Promise((_, reject) =>
        options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true }));
    },
  });
  await assert.rejects(hanging.checkReadiness(), (error) =>
    error.code === "MODEL_NOT_READY" && error.details.reason === "readiness_timeout");
});
