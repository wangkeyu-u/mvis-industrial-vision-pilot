export class RealAcceptanceError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "RealAcceptanceError";
    this.code = code;
  }
}

function textOrNull(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function evaluateRealAcceptanceGate(readiness, { clientMode = "api" } = {}) {
  const runtime = readiness?.runtime && typeof readiness.runtime === "object" ? readiness.runtime : {};
  const activeModel = readiness?.activeModel && typeof readiness.activeModel === "object"
    ? readiness.activeModel : null;
  const selectedMode = textOrNull(runtime.selected_mode) || "unknown";
  const reasons = [];
  if (clientMode !== "api") reasons.push({ code: "API_CLIENT_REQUIRED", message: "浏览器离线 Mock 不能进入真实模型验收。" });
  if (!readiness) reasons.push({ code: "READINESS_MISSING", message: "尚未取得 /health/ready 证据。" });
  else if (readiness.ready !== true) reasons.push({ code: "READINESS_NOT_READY", message: "FastAPI readiness 未通过。" });
  if (selectedMode !== "real") reasons.push({ code: "SELECTED_MODE_NOT_REAL", message: `selected_mode=${selectedMode}，必须为 real。` });
  if (runtime.degraded !== false) reasons.push({
    code: "RUNTIME_DEGRADED",
    message: `运行链路处于降级状态${runtime.fallback_reason ? `：${runtime.fallback_reason}` : "。"}`,
  });
  if (!activeModel) reasons.push({ code: "ACTIVE_MODEL_MISSING", message: "readiness 未返回 active 模型记录。" });
  else {
    if (activeModel.ready !== true) reasons.push({ code: "ACTIVE_MODEL_NOT_READY", message: "active 模型未就绪。" });
    if (activeModel.state && activeModel.state !== "active") {
      reasons.push({ code: "ACTIVE_MODEL_STATE_INVALID", message: `active alias 指向 ${activeModel.state} 生命周期。` });
    }
    if (/mock|test.double|built.in/i.test(`${activeModel.source || ""} ${activeModel.base || ""}`)) {
      reasons.push({ code: "TEST_DOUBLE_DETECTED", message: "模型来源带有 Mock/Test Double 标记。" });
    }
  }
  const allowed = reasons.length === 0;
  return {
    allowed,
    state: allowed ? "verified" : "blocked",
    label: allowed ? "REAL MODEL EVIDENCE" : "REAL SCORE CAPTURE BLOCKED",
    reasons,
    readinessRequestId: textOrNull(readiness?.requestId),
    selectedMode,
    degraded: runtime.degraded ?? null,
    fallbackReason: textOrNull(runtime.fallback_reason),
    model: activeModel ? {
      modelId: textOrNull(activeModel.model_id),
      base: textOrNull(activeModel.base),
      adapter: textOrNull(activeModel.adapter),
      revision: textOrNull(activeModel.revision || activeModel.model_revision),
      state: textOrNull(activeModel.state),
      source: textOrNull(activeModel.source),
      quantization: textOrNull(activeModel.quantization),
      configFingerprint: textOrNull(activeModel.config_fingerprint),
      modelFingerprint: textOrNull(activeModel.model_fingerprint),
      weightHash: textOrNull(activeModel.weight_hash),
    } : null,
  };
}

export function buildRealAcceptanceArtifact({ readiness, clientMode, result = null, evaluation = null, input = null } = {}) {
  const gate = evaluateRealAcceptanceGate(readiness, { clientMode });
  const resultRevision = textOrNull(result?.model?.revision || result?.model?.model_revision);
  const status = gate.allowed ? (result ? "observed" : "ready_no_request") : "blocked";
  return {
    artifact_type: "mvis_real_model_acceptance_gate",
    schema_version: "1.0",
    generated_at: new Date().toISOString(),
    status,
    screenshot_eligible: gate.allowed,
    model_performance_claim_allowed: gate.allowed && evaluation?.eligibleForModelAcceptance === true,
    gate,
    readiness: readiness ? {
      request_id: readiness.requestId || null,
      ready: readiness.ready === true,
      state: readiness.state || null,
      runtime: readiness.runtime || {},
      active_model: readiness.activeModel || null,
    } : null,
    input: input ? {
      sample_id: textOrNull(input.sampleId || input.sample_id),
      license_id: textOrNull(input.licenseId || input.license_id),
      attribution: textOrNull(input.attribution),
      sha256: textOrNull(input.sha256),
      image_bytes_embedded: false,
    } : null,
    observation: result ? {
      request_id: result.request_id || null,
      result: result.policy_state || result.result || null,
      object_count: Array.isArray(result.objects) ? result.objects.length : 0,
      uncertain: Boolean(result.uncertain),
      latency_ms: Number.isFinite(result.latency_ms) ? result.latency_ms : null,
      timing: result.timing || result.latency || null,
      model: result.model || null,
      model_revision: resultRevision || gate.model?.revision || "not_reported",
      image_bytes_embedded: false,
      query_text_embedded: false,
    } : null,
    evaluation: evaluation ? {
      source: evaluation.sourceName || null,
      truth_state: evaluation.truthState || null,
      verified: evaluation.verified === true,
      eligible_for_model_acceptance: evaluation.eligibleForModelAcceptance === true,
    } : null,
  };
}

export function requireRealAcceptance(readiness, options) {
  const gate = evaluateRealAcceptanceGate(readiness, options);
  if (!gate.allowed) {
    throw new RealAcceptanceError(
      "REAL_ACCEPTANCE_BLOCKED",
      gate.reasons.map((reason) => reason.message).join(" ") || "真实模型验收条件未满足。",
    );
  }
  return gate;
}
