const SPECIALIST_SOURCES = new Set([
  "specialist", "florence", "florence2", "rf-detr", "rf_detr", "patchcore",
  "anomaly", "anomaly-map", "anomaly_map", "student-teacher",
]);

function objectValue(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function textValue(...values) {
  return values.find((value) => typeof value === "string" && value.trim())?.trim() || null;
}

function booleanValue(...values) {
  return values.find((value) => typeof value === "boolean") ?? null;
}

function normalizedMode(value) {
  const mode = String(value || "").trim().toLowerCase().replaceAll("-", "_");
  if (["vlm", "vlm_only", "primary_only"].includes(mode)) return "vlm_only";
  if (["specialist", "specialist_only", "anomaly_only"].includes(mode)) return "specialist_only";
  if (["fused", "fusion", "vlm_specialist", "hybrid"].includes(mode)) return "fused";
  return "unreported";
}

function normalizedSource(value) {
  return String(value || "unknown").trim().toLowerCase();
}

function validBbox(value) {
  return Array.isArray(value) && value.length === 4 && value.every(Number.isFinite)
    && value[0] < value[2] && value[1] < value[3];
}

function normalizeRegion(region, fallbackSource) {
  if (!region || typeof region !== "object" || !validBbox(region.bbox)) return null;
  const score = Number.isFinite(region.score) ? region.score
    : Number.isFinite(region.confidence) ? region.confidence : null;
  return {
    bbox: region.bbox.map(Number),
    score: score == null ? null : Math.max(0, Math.min(1, score)),
    source: textValue(region.source, fallbackSource) || "unknown",
  };
}

function normalizeReasons(...values) {
  const flattened = values.flatMap((value) => Array.isArray(value) ? value : value ? [value] : []);
  return [...new Set(flattened.map((value) => typeof value === "string" ? value.trim() : "").filter(Boolean))];
}

function safeHeatmapUrl(value) {
  if (typeof value !== "string" || value.length > 7_000_000) return null;
  if (/^data:image\/(?:png|jpeg|webp);base64,[a-z0-9+/=]+$/i.test(value)) return value;
  return /^\/v1\/artifacts\/heatmaps\/hm_[0-9a-f]{32}$/.test(value) ? value : null;
}

function sourceIsSpecialist(source) {
  const normalized = normalizedSource(source);
  return SPECIALIST_SOURCES.has(normalized) || normalized.startsWith("specialist:");
}

export function normalizeEvidencePresentation(payload, { realRuntime = false } = {}) {
  const trace = objectValue(payload?.trace);
  const provenance = objectValue(payload?.provenance);
  const extra = objectValue(provenance.extra);
  const localization = objectValue(payload?.localization);
  const specialist = objectValue(payload?.specialist || payload?.specialist_result || localization.specialist);
  const fusion = objectValue(payload?.fusion || localization.fusion);
  const vlm = objectValue(payload?.vlm || payload?.primary || localization.vlm);
  const review = objectValue(payload?.human_review || payload?.review);
  const refusal = objectValue(payload?.refusal);
  const evidenceGate = objectValue(payload?.localization_evidence || localization.evidence);
  const objects = Array.isArray(payload?.objects) ? payload.objects : [];
  const sources = [...new Set(objects.map((object) => normalizedSource(object?.source)))];
  let mode = normalizedMode(
    payload?.analysis_mode || payload?.output_mode || trace.execution_mode || trace.mode
      || extra.execution_mode || extra.mode,
  );
  if (mode === "unreported" && objects.length && sources.every((source) => source === "fusion")) mode = "fused";
  if (mode === "unreported" && objects.length && sources.every(sourceIsSpecialist)) mode = "specialist_only";

  const specialistSource = textValue(
    specialist.source, specialist.specialist_id, specialist.model_id, trace.specialist_source, extra.specialist_source,
    sources.find(sourceIsSpecialist),
  );
  const rawHeatmap = objectValue(
    specialist.heatmap || specialist.anomaly_heatmap || payload?.anomaly_heatmap || payload?.heatmap,
  );
  const rawRegions = Array.isArray(rawHeatmap.regions) ? rawHeatmap.regions
    : Array.isArray(rawHeatmap.boxes) ? rawHeatmap.boxes
    : Array.isArray(specialist.heatmap_regions) ? specialist.heatmap_regions
    : Array.isArray(payload?.heatmap_regions) ? payload.heatmap_regions : [];
  const heatmapRegions = rawRegions.map((region) => normalizeRegion(region, specialistSource)).filter(Boolean);
  const heatmapUrl = safeHeatmapUrl(
    rawHeatmap.uri || rawHeatmap.data_url || rawHeatmap.image_data_url || specialist.heatmap_data_url,
  );
  const heatmapAvailable = rawHeatmap.available === true || heatmapRegions.length > 0 || Boolean(heatmapUrl);

  const refusalConflict = String(refusal.code || "").toLowerCase() === "model_conflict";
  const warningConflict = (payload?.warnings || []).some((warning) => /conflict|冲突/i.test(String(warning)));
  const conflictActive = booleanValue(fusion.conflict, payload?.conflict?.active, payload?.conflict) === true
    || refusalConflict || warningConflict;
  const conflictReason = textValue(
    fusion.conflict_reason, fusion.reason, payload?.conflict?.reason,
    refusalConflict ? refusal.message : null,
    conflictActive ? "VLM 与专用定位证据不一致。" : null,
  );

  const reviewRequired = booleanValue(payload?.human_review_required, review.required, refusal.review_required) === true
    || Boolean(payload?.uncertain) || conflictActive;
  const reviewReasons = normalizeReasons(
    review.reasons, review.reason, refusal.review_required ? refusal.message : null,
    conflictActive ? conflictReason : null,
    payload?.uncertain && !conflictActive ? payload?.reason : null,
  );
  if (reviewRequired && !reviewReasons.length) reviewReasons.push("结果不满足自动放行条件，需要人工复核。");

  const warnings = Array.isArray(payload?.warnings) ? payload.warnings.map(String) : [];
  const publicSpecialistEvidence = Boolean(specialist.specialist_id && specialist.revision && specialist.source);
  const backendEvidencePassed = evidenceGate.eligible === true || evidenceGate.verified === true
    || ["verified", "passed"].includes(String(evidenceGate.status || "").toLowerCase())
    || (mode === "specialist_only" && publicSpecialistEvidence)
    || (mode === "fused" && warnings.some((warning) => warning === "fused_evidence_agreement"));
  const specialistReal = booleanValue(
    specialist.real, specialist.is_real, evidenceGate.specialist_real,
  ) === true && !/mock|fixture|test/i.test(`${specialistSource || ""} ${specialist.backend || ""}`)
    || (realRuntime && publicSpecialistEvidence
      && sourceIsSpecialist(specialist.source)
      && !/mock|fixture|test|unavailable/i.test(
        `${specialist.specialist_id || ""} ${specialist.provenance?.backend || ""} ${specialist.revision || ""}`,
      ));
  const fusedAgreement = warnings.some((warning) => warning === "fused_evidence_agreement");
  const fusedGatePassed = booleanValue(fusion.gate_passed, evidenceGate.fusion_gate_passed) === true
    || ["verified", "passed", "corroborated"].includes(String(fusion.status || "").toLowerCase())
    || (mode === "fused" && fusedAgreement && !reviewRequired && !conflictActive && !payload?.uncertain);
  const objectSourcesPass = objects.length > 0 && (mode === "specialist_only"
    ? objects.every((object) => sourceIsSpecialist(object.source))
    : mode === "fused" ? objects.every((object) =>
      sourceIsSpecialist(object.source) || normalizedSource(object.source) === "fusion") : false);
  const localizationVerified = realRuntime && backendEvidencePassed && objectSourcesPass
    && (mode === "specialist_only" ? specialistReal : mode === "fused" ? specialistReal && fusedGatePassed : false);
  const localizationReason = localizationVerified
    ? mode === "fused" ? "真实 specialist 与 VLM 证据通过融合门禁。" : "真实 specialist 输出已通过定位证据门禁。"
    : textValue(
      evidenceGate.reason,
      !realRuntime ? "当前结果未绑定真实、非降级 readiness。" : null,
      !backendEvidencePassed ? "后端未明确声明定位证据验收通过。" : null,
      !objectSourcesPass ? "框来源不是合格的 specialist_only/fused 输出。" : null,
      mode === "fused" && !fusedGatePassed ? "fused 结果未通过一致性门禁。" : null,
      "定位证据未通过真实性门禁。",
    );

  return {
    mode,
    modeLabel: ({ vlm_only: "VLM ONLY", specialist_only: "SPECIALIST ONLY", fused: "FUSED" })[mode] || "MODE UNREPORTED",
    vlmExplanation: textValue(
      vlm.reason, vlm.explanation, payload?.vlm_explanation,
      mode === "vlm_only" ? payload?.reason : null,
      mode === "fused" ? "后端未单独返回 VLM 原始解释；当前仅有融合结论。" : null,
    ),
    specialist: {
      source: specialistSource,
      modelId: textValue(specialist.specialist_id, specialist.model_id, specialist.model),
      revision: textValue(specialist.revision, specialist.provenance?.checkpoint_revision),
      score: Number.isFinite(specialist.score) ? specialist.score : null,
      threshold: Number.isFinite(specialist.threshold) ? specialist.threshold : null,
      qualityStatus: textValue(specialist.quality_status),
      qualityAccepted: specialist.quality_accepted === true,
      real: specialistReal,
      objectCount: Array.isArray(specialist.objects) ? specialist.objects.length
        : objects.filter((object) => sourceIsSpecialist(object.source) || normalizedSource(object.source) === "fusion").length,
      heatmapAvailable,
      heatmapRegions,
      heatmapUrl,
    },
    fusion: {
      status: textValue(
        fusion.status,
        mode === "fused" ? fusedAgreement ? "agreement" : conflictActive ? "conflict" : "unreported" : "not_applicable",
      ),
      gatePassed: fusedGatePassed,
      conflict: conflictActive,
      reason: conflictReason,
    },
    review: { required: reviewRequired, reasons: reviewReasons },
    localization: {
      verified: localizationVerified,
      status: localizationVerified ? "verified" : objects.length ? "blocked" : "no_evidence",
      reason: localizationReason,
      backendEvidencePassed,
      objectSourcesPass,
    },
  };
}

export const evidencePresentationContract = Object.freeze({
  modes: ["vlm_only", "specialist_only", "fused"],
  specialistSources: [...SPECIALIST_SOURCES],
});
