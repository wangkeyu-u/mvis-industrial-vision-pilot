const ACCEPTED_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

export class AnalysisClientError extends Error {
  constructor(code, message, { status = 0, requestId = null, details = null } = {}) {
    super(message);
    this.name = "AnalysisClientError";
    this.code = code;
    this.status = status;
    this.requestId = requestId;
    this.details = details;
  }
}

export async function validateImageFile(file) {
  if (!file || typeof file.arrayBuffer !== "function") {
    throw new AnalysisClientError("INVALID_IMAGE", "请选择一张 JPG、PNG 或 WEBP 图片。");
  }
  if (!ACCEPTED_MIME_TYPES.has(file.type)) {
    throw new AnalysisClientError("INVALID_IMAGE", "仅支持 JPG、PNG、WEBP 图片。");
  }
  if (!file.size || file.size > MAX_IMAGE_BYTES) {
    throw new AnalysisClientError("INVALID_IMAGE", "图片必须大于 0 且不超过 10 MB。");
  }
  const header = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  const matches =
    (file.type === "image/jpeg" && header[0] === 0xff && header[1] === 0xd8 && header[2] === 0xff) ||
    (file.type === "image/png" && [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a].every((v, i) => header[i] === v)) ||
    (file.type === "image/webp" && ascii(header, 0, 4) === "RIFF" && ascii(header, 8, 12) === "WEBP");
  if (!matches) {
    throw new AnalysisClientError("INVALID_IMAGE", "图片 MIME 与文件签名不一致。");
  }
  return file;
}

function ascii(bytes, start, end) {
  return String.fromCharCode(...bytes.slice(start, end));
}

function validateRequest(request) {
  const query = request?.query?.trim();
  if (!query || query.length > 1000) {
    throw new AnalysisClientError("INVALID_QUERY", "查询必须为 1–1000 个字符。");
  }
  return query;
}

function requestId() {
  const suffix = globalThis.crypto?.randomUUID?.().slice(0, 8) ?? Math.random().toString(16).slice(2, 10);
  return `req_demo_${Date.now()}_${suffix}`;
}

function apiRequestId() {
  const suffix = globalThis.crypto?.randomUUID?.().replaceAll("-", "").slice(0, 16)
    ?? Math.random().toString(16).slice(2, 18);
  return `req_ui_${Date.now()}_${suffix}`;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function buildMockResult(request) {
  const query = request.query.trim().toLowerCase();
  const width = Math.max(1, Number(request.imageWidth) || 1280);
  const height = Math.max(1, Number(request.imageHeight) || 800);
  let result = "violation";
  let uncertain = false;
  let objects = [{
    label: "未佩戴防护装备",
    bbox: [Math.round(width * 0.56), Math.round(height * 0.18), Math.round(width * 0.82), Math.round(height * 0.72)],
    confidence: 0.91,
    source: request.useSpecialist ? "fusion" : "vlm",
  }];
  let reason = "检测到人员作业区域存在防护装备缺失，请人工复核标注区域。";
  const warnings = ["当前为可替换的演示 Mock，结果不得用于业务决策。"];

  if (["拒答", "高风险", "身份证", "人脸识别", "medical"].some((token) => query.includes(token))) {
    result = "refused";
    uncertain = true;
    objects = [];
    reason = "该请求属于高风险或超出系统用途范围，已拒绝自动判断，请交由人工处理。";
    warnings.push("系统不会对高风险场景给出确定性结论。");
  } else if (["不确定", "证据不足", "遮挡", "uncertain", "模糊"].some((token) => query.includes(token))) {
    result = "uncertain";
    uncertain = true;
    objects = [];
    reason = "图像证据不足，无法形成可靠结论；建议补充更清晰、无遮挡的图片。";
    warnings.push("置信度低于策略阈值，未生成边界框。");
  } else if (["未发现", "通过", "compliant", "合规样例"].some((token) => query.includes(token))) {
    result = "compliant";
    objects = [];
    reason = "在当前图片与查询范围内未发现明确违规证据。";
  }

  return {
    schema_version: "1.0",
    request_id: requestId(),
    model: {
      base: "qwen3-vl-2b-instruct-4bit",
      adapter: "compliance-lora-v0.1-demo",
      alias: request.model || "active",
      quantization: "4-bit",
    },
    result,
    objects,
    reason,
    uncertain,
    latency_ms: 386,
    timing: { preprocess_ms: 42, inference_ms: 311, validation_ms: 33 },
    warnings,
    trace: {
      task: request.task || "inspect",
      use_specialist: Boolean(request.useSpecialist),
      source: "browser-mock",
      image_persisted: false,
    },
  };
}

export class MockAnalysisClient {
  constructor({ latencyMs = 450 } = {}) {
    this.latencyMs = latencyMs;
    this.kind = "mock";
  }

  async analyze(request, { signal } = {}) {
    request.query = validateRequest(request);
    await validateImageFile(request.image);
    if (["timeout", "超时", "错误演示"].some((token) => request.query.toLowerCase().includes(token))) {
      await delay(Math.min(this.latencyMs, 80));
      throw new AnalysisClientError("INFERENCE_TIMEOUT", "模拟推理超时；请重试或降低输入复杂度。", { status: 504 });
    }
    await Promise.race([
      delay(this.latencyMs),
      new Promise((_, reject) => signal?.addEventListener("abort", () => reject(new DOMException("请求已取消", "AbortError")), { once: true })),
    ]);
    return buildMockResult(request);
  }

  async checkReadiness() {
    return {
      ready: true,
      state: "mock",
      label: "离线 Mock",
      detail: "浏览器内置替身·未连接 FastAPI",
      requestId: null,
      runtime: { requested_mode: "mock", selected_mode: "browser-mock", degraded: true },
      activeModel: null,
      aliases: {},
      models: [],
    };
  }
}

async function fileToDataUrl(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const chunk = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunk) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunk));
  }
  return `data:${file.type};base64,${btoa(binary)}`;
}

export function normalizeAnalysisResponse(payload) {
  const validObjects = Array.isArray(payload?.objects) && payload.objects.every((object) =>
    object && typeof object === "object"
    && typeof object.label === "string" && object.label.length > 0
    && Array.isArray(object.bbox) && object.bbox.length === 4
    && object.bbox.every(Number.isFinite)
    && object.bbox[0] < object.bbox[2] && object.bbox[1] < object.bbox[3]
    && Number.isFinite(object.confidence) && object.confidence >= 0 && object.confidence <= 1
    && typeof object.source === "string" && object.source.length > 0
  );
  const valid = payload && typeof payload === "object"
    && typeof payload.request_id === "string"
    && payload.request_id.length > 0
    && payload.model && typeof payload.model === "object" && typeof payload.model.base === "string"
    && typeof payload.result === "string"
    && validObjects
    && typeof payload.reason === "string";
  if (!valid) {
    throw new AnalysisClientError(
      "OUTPUT_VALIDATION_FAILED",
      "服务响应缺少必需字段，无法安全展示。",
      { status: 422, requestId: payload?.request_id || null },
    );
  }
  const timing = payload.timing || payload.latency || {};
  const warnings = Array.isArray(payload.warnings) ? payload.warnings : [];
  const policyState = warnings.some((warning) => String(warning).toLowerCase().startsWith("refusal:"))
    ? "refused"
    : payload.result;
  return {
    ...payload,
    policy_state: policyState,
    uncertain: Boolean(payload.uncertain),
    warnings,
    timing,
    latency_ms: Number.isFinite(payload.latency_ms)
      ? payload.latency_ms
      : [timing.preprocess_ms, timing.inference_ms, timing.validation_ms]
        .reduce((total, value) => total + (Number(value) || 0), 0),
  };
}

function requestSignal(externalSignal, timeoutMs) {
  const controller = new AbortController();
  const state = { timedOut: false };
  const abortFromCaller = () => controller.abort(externalSignal?.reason || new DOMException("请求已取消", "AbortError"));
  if (externalSignal?.aborted) abortFromCaller();
  else externalSignal?.addEventListener("abort", abortFromCaller, { once: true });
  const timer = setTimeout(() => {
    state.timedOut = true;
    controller.abort(new DOMException("请求超时", "TimeoutError"));
  }, timeoutMs);
  return {
    signal: controller.signal,
    state,
    cleanup() {
      clearTimeout(timer);
      externalSignal?.removeEventListener("abort", abortFromCaller);
    },
  };
}

export function classifyReadiness(payload) {
  const status = payload?.status;
  const details = payload?.details && typeof payload.details === "object" ? payload.details : {};
  const runtime = details.runtime && typeof details.runtime === "object" ? details.runtime : {};
  const aliases = details.aliases && typeof details.aliases === "object" ? details.aliases : {};
  const models = Array.isArray(details.models) ? details.models : [];
  const activeModel = models.find((model) => model?.model_id === aliases.active) || null;
  const selectedMode = String(runtime.selected_mode || "unknown");
  const source = String(activeModel?.source || "");
  const modelLooksLikeTestDouble = /mock|test.double|built.in/i.test(`${selectedMode} ${source}`);
  if (status !== "ready" || payload?.model_ready === false || activeModel?.ready === false || !activeModel) {
    return {
      ready: false,
      state: "unavailable",
      label: "FastAPI 不可用",
      detail: status === "not_ready" ? "服务存活，但模型未就绪" : "readiness 预检未通过",
      requestId: payload?.request_id || null,
      runtime,
      activeModel,
      aliases,
      models,
    };
  }
  if (runtime.degraded === true || modelLooksLikeTestDouble) {
    return {
      ready: true,
      state: "degraded",
      label: "FastAPI 就绪 · 模型替身",
      detail: runtime.fallback_reason || `${selectedMode} 降级链路`,
      requestId: payload?.request_id || null,
      runtime,
      activeModel,
      aliases,
      models,
    };
  }
  return {
    ready: true,
    state: "real",
    label: "真实模型就绪",
    detail: `${activeModel?.base || selectedMode} · FastAPI`,
    requestId: payload?.request_id || null,
    runtime,
    activeModel,
    aliases,
    models,
  };
}

async function buildTransport(request, query, transport) {
  if (transport === "multipart") {
    const body = new FormData();
    body.append("image", request.image, request.image.name || "upload-image");
    body.append("query", query);
    body.append("task", request.task || "inspect");
    body.append("model", request.model || "active");
    body.append("use_specialist", String(Boolean(request.useSpecialist)));
    body.append("options", JSON.stringify({ temperature: 0, seed: 42 }));
    return { body, headers: {} };
  }
  return {
    body: JSON.stringify({
      image: await fileToDataUrl(request.image),
      image_width: request.imageWidth,
      image_height: request.imageHeight,
      query,
      task: request.task || "inspect",
      model: request.model || "active",
      use_specialist: Boolean(request.useSpecialist),
      options: { temperature: 0, seed: 42 },
    }),
    headers: { "Content-Type": "application/json" },
  };
}

export class HttpAnalysisClient {
  constructor({
    endpoint = "/v1/analyze",
    readinessEndpoint = "/health/ready",
    transport = "json",
    requestTimeoutMs = 10000,
    fetchImpl = globalThis.fetch,
  } = {}) {
    if (typeof fetchImpl !== "function") throw new TypeError("HttpAnalysisClient requires fetch.");
    if (!["json", "multipart"].includes(transport)) throw new TypeError("transport must be json or multipart.");
    if (!Number.isFinite(requestTimeoutMs) || requestTimeoutMs <= 0) throw new TypeError("requestTimeoutMs must be positive.");
    this.endpoint = endpoint;
    this.readinessEndpoint = readinessEndpoint;
    this.transport = transport;
    this.requestTimeoutMs = requestTimeoutMs;
    // Browser-native fetch requires its host global as `this` in some engines.
    this.fetchImpl = fetchImpl.bind(globalThis);
    this.kind = "api";
  }

  async checkReadiness({ signal: externalSignal } = {}) {
    const requestIdValue = apiRequestId();
    const scopedSignal = requestSignal(externalSignal, this.requestTimeoutMs);
    let response;
    try {
      response = await this.fetchImpl(this.readinessEndpoint, {
        method: "GET",
        headers: { Accept: "application/json", "X-Request-ID": requestIdValue },
        signal: scopedSignal.signal,
      });
    } catch (error) {
      if (externalSignal?.aborted || (error?.name === "AbortError" && !scopedSignal.state.timedOut)) {
        throw new DOMException("请求已取消", "AbortError");
      }
      throw new AnalysisClientError("MODEL_NOT_READY", scopedSignal.state.timedOut
        ? `readiness 预检超过 ${this.requestTimeoutMs} ms。`
        : "无法连接 FastAPI readiness 端点。", {
        status: 503,
        requestId: requestIdValue,
        details: { reason: scopedSignal.state.timedOut ? "readiness_timeout" : "network_unavailable" },
      });
    } finally {
      scopedSignal.cleanup();
    }
    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new AnalysisClientError("MODEL_NOT_READY", "readiness 返回了无法解析的响应。", {
        status: response.status,
        requestId: response.headers?.get?.("x-request-id") || requestIdValue,
      });
    }
    const classified = classifyReadiness(payload);
    if (!response.ok || !classified.ready) {
      throw new AnalysisClientError(payload?.error?.code || "MODEL_NOT_READY", payload?.error?.message || classified.detail, {
        status: response.status,
        requestId: classified.requestId || payload?.error?.request_id || response.headers?.get?.("x-request-id") || requestIdValue,
        details: payload?.details || payload?.error?.details || null,
      });
    }
    return classified;
  }

  async analyze(request, { signal: externalSignal } = {}) {
    const query = validateRequest(request);
    await validateImageFile(request.image);
    const requestIdValue = apiRequestId();
    const transport = await buildTransport(request, query, this.transport);
    const scopedSignal = requestSignal(externalSignal, this.requestTimeoutMs);
    let response;
    try {
      response = await this.fetchImpl(this.endpoint, {
        method: "POST",
        headers: { ...transport.headers, Accept: "application/json", "X-Request-ID": requestIdValue },
        signal: scopedSignal.signal,
        body: transport.body,
      });
    } catch (error) {
      if (scopedSignal.state.timedOut) {
        throw new AnalysisClientError("INFERENCE_TIMEOUT", `请求超过 ${this.requestTimeoutMs} ms，已取消。`, {
          status: 504,
          requestId: requestIdValue,
        });
      }
      if (externalSignal?.aborted || error?.name === "AbortError") throw new DOMException("请求已取消", "AbortError");
      throw new AnalysisClientError("MODEL_NOT_READY", "无法连接 FastAPI 服务，请检查后端与同源代理。", {
        status: 503,
        requestId: requestIdValue,
      });
    } finally {
      scopedSignal.cleanup();
    }
    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new AnalysisClientError("INTERNAL_ERROR", "服务返回了无法解析的响应。", { status: response.status });
    }
    if (!response.ok) {
      throw new AnalysisClientError(payload.error?.code || "INTERNAL_ERROR", payload.error?.message || "分析失败。", {
        status: response.status,
        requestId: payload.request_id || payload.error?.request_id || response.headers?.get?.("x-request-id") || requestIdValue,
        details: payload.error?.details,
      });
    }
    const normalized = normalizeAnalysisResponse(payload);
    return {
      ...normalized,
      trace: normalized.trace || {
        source: response.headers?.get?.("x-mvis-backend") || "fastapi",
        transport: this.transport,
      },
    };
  }
}

export function createAnalysisClient({
  mode = "mock",
  endpoint = "/v1/analyze",
  readinessEndpoint = "/health/ready",
  transport = "json",
  requestTimeoutMs = 10000,
  fetchImpl,
  latencyMs,
} = {}) {
  return mode === "api"
    ? new HttpAnalysisClient({ endpoint, readinessEndpoint, transport, requestTimeoutMs, fetchImpl })
    : new MockAnalysisClient({ latencyMs });
}

export const clientLimits = Object.freeze({
  acceptedMimeTypes: [...ACCEPTED_MIME_TYPES],
  maxImageBytes: MAX_IMAGE_BYTES,
  maxQueryCharacters: 1000,
});
