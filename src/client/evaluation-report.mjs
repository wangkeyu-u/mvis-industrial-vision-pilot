const MAX_FILE_BYTES = 5 * 1024 * 1024;
const MAX_PACKAGE_FILES = 16;

const pilotMetricTargets = Object.freeze({
  macro_f1: { minimum: 0.82 },
  acc_at_iou: { minimum: 0.70 },
  json_schema_validity_rate: { minimum: 0.99 },
  hard_negative_false_positive_rate: { maximum: 0.10 },
  evidence_conclusion_consistency_rate: { minimum: 0.97 },
});

export const evaluationMetricDefinitions = Object.freeze([
  { key: "macro_f1", label: "Macro-F1", direction: "higher" },
  { key: "acc_at_iou", label: "Acc@IoU 0.5", direction: "higher" },
  { key: "json_schema_validity_rate", label: "JSON 有效率", direction: "higher" },
  { key: "hard_negative_false_positive_rate", label: "困难负例 FPR", direction: "lower" },
  { key: "evidence_conclusion_consistency_rate", label: "证据—结论一致率", direction: "higher" },
]);

const REQUIRED_REPORT_KEYS = ["schema_version", "summary", "slice_metrics", "samples", "failure_cases"];

export class EvaluationImportError extends Error {
  constructor(code, message, details = null) {
    super(message);
    this.name = "EvaluationImportError";
    this.code = code;
    this.details = details;
  }
}

function objectValue(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new EvaluationImportError("INVALID_EVALUATION_JSON", `${label} 必须是 JSON object。`);
  }
  return value;
}

function finiteMetric(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new EvaluationImportError("INVALID_METRIC", `${label} 必须是 0–1 之间的有限数字。`);
  }
  return value;
}

function finiteDelta(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < -1 || value > 1) {
    throw new EvaluationImportError("INVALID_METRIC", `${label} 必须是 -1–1 之间的有限数字。`);
  }
  return value;
}

function nonNegativeCount(value, label) {
  if (!Number.isInteger(value) || value < 0) {
    throw new EvaluationImportError("INVALID_METRIC", `${label} 必须是非负整数。`);
  }
  return value;
}

function normalizeSummary(raw) {
  const summary = objectValue(raw, "summary");
  const normalized = {};
  for (const metric of evaluationMetricDefinitions) {
    normalized[metric.key] = finiteMetric(summary[metric.key], metric.key);
  }
  for (const key of ["sample_count", "localization_target_count", "hard_negative_count", "failure_case_count"]) {
    normalized[key] = nonNegativeCount(summary[key], key);
  }
  normalized.iou_threshold = typeof summary.iou_threshold === "number" ? finiteMetric(summary.iou_threshold, "iou_threshold") : 0.5;
  return normalized;
}

function normalizeReport(raw) {
  const report = objectValue(raw, "report");
  const missing = REQUIRED_REPORT_KEYS.filter((key) => !(key in report));
  if (missing.length) {
    throw new EvaluationImportError("INVALID_REPORT", `report 缺少字段：${missing.join(", ")}。`);
  }
  if (report.schema_version !== "1.0.0") {
    throw new EvaluationImportError("UNSUPPORTED_SCHEMA", `不支持的 report schema：${report.schema_version ?? "missing"}。`);
  }
  if (!Array.isArray(report.samples) || !Array.isArray(report.failure_cases)) {
    throw new EvaluationImportError("INVALID_REPORT", "samples 和 failure_cases 必须是数组。");
  }
  const sampleIds = report.samples.map((sample) => sample?.sample_id);
  if (sampleIds.some((sampleId) => typeof sampleId !== "string" || !sampleId.trim())) {
    throw new EvaluationImportError("INVALID_REPORT", "report.samples 中每条记录都必须包含 sample_id。");
  }
  if (new Set(sampleIds).size !== sampleIds.length) {
    throw new EvaluationImportError("DUPLICATE_SAMPLE_ID", "report.samples 含重复 sample_id，不能用于配对比较。");
  }
  objectValue(report.slice_metrics, "slice_metrics");
  return report;
}

function normalizedInterval(interval) {
  if (!interval || typeof interval !== "object") return null;
  const lower = interval.lower;
  const upper = interval.upper;
  if (lower !== null && lower !== undefined) finiteMetric(lower, "confidence lower");
  if (upper !== null && upper !== undefined) finiteMetric(upper, "confidence upper");
  if (lower != null && upper != null && lower > upper) {
    throw new EvaluationImportError("INVALID_INTERVAL", "置信区间 lower 不能大于 upper。");
  }
  return {
    lower: lower ?? null,
    upper: upper ?? null,
    level: typeof interval.confidence_level === "number" ? interval.confidence_level : null,
    observationCount: Number.isInteger(interval.observation_count) ? interval.observation_count : null,
    method: typeof interval.method === "string" ? interval.method : null,
  };
}

function normalizeFailureCase(sample) {
  const value = objectValue(sample, "failure case");
  if (typeof value.sample_id !== "string" || !value.sample_id) {
    throw new EvaluationImportError("INVALID_FAILURE_CASE", "失败案例缺少 sample_id。");
  }
  if (!Array.isArray(value.failure_codes)) {
    throw new EvaluationImportError("INVALID_FAILURE_CASE", `${value.sample_id} 的 failure_codes 必须是数组。`);
  }
  return {
    sampleId: value.sample_id,
    codes: value.failure_codes.map(String),
    truth: typeof value.truth?.result === "string" ? value.truth.result : "unknown",
    prediction: typeof value.prediction?.result === "string" ? value.prediction.result : "invalid/missing",
    source: typeof value.prediction?.source_kind === "string" ? value.prediction.source_kind : "unknown",
    metrics: value.metrics && typeof value.metrics === "object" ? value.metrics : {},
  };
}

function normalizeSlices(report, failures) {
  if (!report) return [];
  const failureSlices = failures?.by_slice && typeof failures.by_slice === "object" ? failures.by_slice : {};
  return Object.entries(report.slice_metrics).map(([name, raw]) => {
    const summary = normalizeSummary(raw);
    const failure = failureSlices[name] && typeof failureSlices[name] === "object" ? failureSlices[name] : {};
    const failureCount = Number.isInteger(failure.failure_count) ? failure.failure_count : summary.failure_case_count;
    return {
      name,
      ...summary,
      failureCount,
      failureRate: typeof failure.failure_rate === "number"
        ? finiteMetric(failure.failure_rate, `${name} failure_rate`)
        : summary.sample_count ? failureCount / summary.sample_count : 0,
      failureCodes: failure.failure_codes && typeof failure.failure_codes === "object" ? failure.failure_codes : {},
    };
  }).sort((left, right) => right.failureRate - left.failureRate || right.sample_count - left.sample_count || left.name.localeCompare(right.name));
}

function assessmentMap(metrics) {
  const items = metrics?.kpi_acceptance?.assessments;
  if (!Array.isArray(items)) return new Map();
  return new Map(items.filter((item) => item && typeof item.metric === "string").map((item) => [item.metric, item]));
}

function fixtureTruth(manifest, metrics, config) {
  return manifest?.fixture_only === true || metrics?.fixture_only === true || config?.fixture_only === true;
}

function mockTruth(manifest, metrics, config) {
  return manifest?.mock_only === true || metrics?.mock_only === true || config?.mock_only === true;
}

function pilotTruth(manifest, metrics, config, provenance) {
  return manifest?.pilot_only === true || metrics?.pilot_only === true || config?.pilot_only === true
    || provenance?.formal_kpi_eligible === false || provenance?.evaluation_status === "pilot";
}

function stableValue(value) {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableValue(value[key])]));
  }
  return value;
}

function evidenceValue(value) {
  if (value && typeof value === "object" && !Array.isArray(value) && "status" in value && "value" in value) {
    return value.status === "recorded" ? value.value : null;
  }
  return value ?? null;
}

function findNamedValues(root, names, results = [], seen = new Set()) {
  if (!root || typeof root !== "object" || seen.has(root)) return results;
  seen.add(root);
  for (const [key, raw] of Object.entries(root)) {
    if (names.has(key)) results.push(evidenceValue(raw));
    if (raw && typeof raw === "object") findNamedValues(raw, names, results, seen);
  }
  return results;
}

function uniqueTextIdentity(root, names, label) {
  const values = findNamedValues(root, new Set(names))
    .filter((value) => typeof value === "string" && value.trim())
    .map((value) => value.trim());
  const unique = [...new Set(values)];
  if (unique.length > 1) {
    throw new EvaluationImportError("MIXED_RUN_IDENTITY", `${label} 在包内存在多个值，不能作为同一次运行证据。`, unique);
  }
  return unique[0] || null;
}

function firstNumber(root, names) {
  return findNamedValues(root, new Set(names)).find((value) => typeof value === "number" && Number.isFinite(value)) ?? null;
}

function promptIdentity(...sources) {
  const roots = sources.filter((source) => source && typeof source === "object");
  if (!roots.length) return { identity: null, display: null };
  const searchable = Object.fromEntries(roots.map((source, index) => [`source_${index}`, source]));
  const explicit = uniqueTextIdentity(
    searchable,
    ["prompt_sha256", "test_prompt_sha256", "prompt_query_sha256", "prompt_hash", "prompt_fingerprint"],
    "prompt hash",
  );
  if (explicit) return { identity: explicit, display: explicit };
  const prompt = roots.map((source) => source.prompt ?? source.prompt_template ?? source.system_prompt)
    .find((value) => value != null) ?? null;
  if (prompt == null) {
    const version = uniqueTextIdentity(searchable, ["prompt_version", "prompt_id"], "prompt version");
    return { identity: version, display: version };
  }
  const canonical = JSON.stringify(stableValue(prompt));
  return {
    identity: canonical,
    display: typeof prompt === "string" ? prompt : canonical,
  };
}

function absolutePilotPass(metric) {
  const target = pilotMetricTargets[metric.key] || {};
  return !(typeof target.minimum === "number" && metric.candidate < target.minimum)
    && !(typeof target.maximum === "number" && metric.candidate > target.maximum);
}

function normalizeRunEvidence(bundle, report) {
  const prompt = promptIdentity(bundle.config, bundle.run_manifest, bundle.experiment);
  const searchable = {
    config: bundle.config,
    environment: bundle.environment,
    run_manifest: bundle.run_manifest,
    experiment: bundle.experiment,
    performance: bundle.performance,
    sample_predictions: report?.samples?.map((sample) => sample?.prediction).filter(Boolean),
  };
  const directRevision = [
    bundle.config?.model_revision,
    bundle.config?.base_model_revision,
    bundle.config?.checkpoint_revision,
    bundle.config?.revision,
    bundle.config?.model?.revision,
    bundle.config?.run?.base_model_revision,
    bundle.run_manifest?.model?.revision,
    bundle.run_manifest?.artifacts?.checkpoint_sha256,
    bundle.run_manifest?.configuration?.backbone_revision,
  ].map(evidenceValue).find((value) => typeof value === "string" && value.trim()) || null;
  const directAdapterHash = [
    bundle.config?.adapter_hash,
    bundle.config?.adapter_sha256,
    bundle.config?.run?.adapter?.enabled === true ? bundle.config?.run?.adapter?.sha256 : null,
    bundle.run_manifest?.adapter?.enabled === true ? bundle.run_manifest?.adapter?.sha256 : null,
    bundle.run_manifest?.model?.adapter_hash,
    bundle.run_manifest?.model?.adapter_sha256,
  ].map(evidenceValue).find((value) => typeof value === "string" && value.trim()) || null;
  const modelId = uniqueTextIdentity(searchable, ["base_model_id"], "base model id")
    || uniqueTextIdentity(searchable, ["model_id"], "model id");
  const adapterHash = uniqueTextIdentity(searchable, ["adapter_hash", "adapter_sha256", "adapter_fingerprint"], "adapter hash")
    || (typeof directAdapterHash === "string" ? directAdapterHash.trim() : null);
  const declaredKind = textIdentity([
    bundle.config?.candidate_kind,
    bundle.config?.experiment_kind,
    bundle.config?.run?.candidate_kind,
    bundle.config?.run?.kind,
    bundle.experiment?.kind,
    bundle.run_manifest?.candidate_kind,
    bundle.run_manifest?.experiment_kind,
    bundle.run_manifest?.run_kind,
    bundle.run_manifest?.configuration?.algorithm,
    bundle.run_manifest?.implementation?.algorithm,
    report?.samples?.[0]?.prediction?.model_id,
  ]);
  const candidateKind = /specialist|detector|anomaly|patchcore/.test(declaredKind || "") ? "specialist"
    : /fusion|fused|hybrid/.test(declaredKind || "") ? "fused"
    : /lora|qlora|adapter/.test(declaredKind || "") || adapterHash ? "lora"
    : /zero.?shot|baseline/.test(declaredKind || "") ? "zero_shot" : "unknown";
  const specialistIdentity = uniqueTextIdentity(
    searchable,
    ["specialist_model_id", "specialist_id", "detector_id", "anomaly_model_id"],
    "specialist identity",
  ) || (candidateKind === "specialist" ? modelId : null);
  return {
    modelId,
    modelRevision: uniqueTextIdentity(searchable, ["model_revision", "base_model_revision", "checkpoint_revision"], "model revision")
      || (typeof directRevision === "string" ? directRevision.trim() : null),
    adapterHash,
    candidateKind,
    specialistIdentity,
    promptIdentity: prompt.identity,
    promptDisplay: prompt.display,
    runId: uniqueTextIdentity(searchable, ["run_id"], "run id")
      || uniqueTextIdentity(searchable, ["experiment_id"], "experiment id"),
    qualityStatus: uniqueTextIdentity(searchable, ["quality_status"], "quality status"),
    resources: {
      p50LatencyMs: firstNumber(searchable, ["p50_latency_ms", "latency_p50_ms"]),
      p95LatencyMs: firstNumber(searchable, ["p95_latency_ms", "latency_p95_ms"]),
      peakMemoryMb: firstNumber(searchable, ["peak_memory_mb", "process_peak_memory_after_mb", "process_peak_rss_mb", "mlx_peak_allocated_mb"]),
      processPeakRssMb: firstNumber(searchable, ["process_peak_rss_mb", "process_peak_memory_after_mb"]),
      mlxPeakAllocatedMb: firstNumber(searchable, ["mlx_peak_allocated_mb"]),
    },
  };
}

function textIdentity(values) {
  return values.find((value) => typeof value === "string" && value.trim())?.trim().toLowerCase() || null;
}

function buildPortfolio(bundle, { sourceName = "evaluation.json", verified = false, kind = "package" } = {}) {
  const manifest = bundle.package_manifest ? objectValue(bundle.package_manifest, "package_manifest") : null;
  const metrics = bundle.metrics ? objectValue(bundle.metrics, "metrics") : null;
  const report = bundle.report ? normalizeReport(bundle.report) : null;
  if (!metrics && !report) {
    throw new EvaluationImportError("INVALID_EVALUATION_JSON", "导入内容必须包含 report 或 metrics。");
  }
  const summary = normalizeSummary(metrics?.summary || report?.summary);
  if (report) {
    const reportSummary = normalizeSummary(report.summary);
    if (report.samples.length > 0 && report.samples.length !== summary.sample_count) {
      throw new EvaluationImportError(
        "SAMPLE_COUNT_MISMATCH",
        `report.samples 有 ${report.samples.length} 条，但 summary.sample_count=${summary.sample_count}。`,
      );
    }
    for (const metric of evaluationMetricDefinitions) {
      if (Math.abs(reportSummary[metric.key] - summary[metric.key]) > 1e-12) {
        throw new EvaluationImportError("METRICS_REPORT_MISMATCH", `metrics 与 report 的 ${metric.key} 不一致。`);
      }
    }
  }
  const assessments = assessmentMap(metrics);
  const pairedComparison = bundle.comparison && typeof bundle.comparison === "object" ? bundle.comparison : null;
  if (pairedComparison && pairedComparison.fair_comparison !== true) {
    throw new EvaluationImportError("UNFAIR_COMPARISON", "comparison.json 未通过配对总体一致性检查。");
  }
  const intervals = metrics?.confidence_intervals && typeof metrics.confidence_intervals === "object"
    ? metrics.confidence_intervals : {};
  const fixtureOnly = fixtureTruth(manifest, metrics, bundle.config);
  const mockOnly = mockTruth(manifest, metrics, bundle.config);
  const pilotOnly = pilotTruth(manifest, metrics, bundle.config, bundle.data_provenance);
  const acceptance = metrics?.kpi_acceptance && typeof metrics.kpi_acceptance === "object"
    ? metrics.kpi_acceptance : { status: "not_evaluable", eligible_for_model_acceptance: false, assessments: [] };
  const eligible = verified && !fixtureOnly && !mockOnly && !pilotOnly
    && acceptance.eligible_for_model_acceptance === true;
  const truthState = fixtureOnly ? "fixture" : mockOnly ? "mock" : pilotOnly && verified ? "pilot" : verified ? "verified" : "unverified";
  const watermark = fixtureOnly
    ? "FIXTURE / MOCK · 不可用于模型验收"
    : mockOnly ? "MOCK OUTPUT · 不可用于模型验收"
    : pilotOnly && verified ? `PILOT ONLY · ${summary.sample_count} TEST SAMPLES · 禁止包装为正式 KPI`
    : verified ? null : "UNVERIFIED REPORT · 不可用于模型验收";
  const comparison = evaluationMetricDefinitions.map((definition) => {
    const assessment = assessments.get(definition.key);
    const paired = pairedComparison?.metrics?.[definition.key];
    let pairedCandidate = null;
    let pairedDelta = null;
    let improvementInterval = null;
    if (paired) {
      pairedCandidate = finiteMetric(paired.candidate, `${definition.key} candidate`);
      if (Math.abs(pairedCandidate - summary[definition.key]) > 1e-12) {
        throw new EvaluationImportError("COMPARISON_REPORT_MISMATCH", `comparison 与 report 的 ${definition.key} 不一致。`);
      }
      pairedDelta = finiteDelta(paired.delta, `${definition.key} delta`);
      const lower = paired.improvement_ci_lower == null
        ? null : finiteDelta(paired.improvement_ci_lower, `${definition.key} improvement_ci_lower`);
      const upper = paired.improvement_ci_upper == null
        ? null : finiteDelta(paired.improvement_ci_upper, `${definition.key} improvement_ci_upper`);
      if (lower != null && upper != null && lower > upper) {
        throw new EvaluationImportError("INVALID_INTERVAL", `${definition.key} 的配对改善区间 lower 不能大于 upper。`);
      }
      improvementInterval = {
        lower,
        upper,
        observationCount: paired.observation_count == null
          ? null : nonNegativeCount(paired.observation_count, `${definition.key} observation_count`),
      };
    }
    const baseline = typeof paired?.baseline === "number"
      ? finiteMetric(paired.baseline, `${definition.key} baseline`)
      : typeof assessment?.baseline === "number" ? assessment.baseline : null;
    return {
      ...definition,
      candidate: summary[definition.key],
      baseline,
      delta: paired ? pairedDelta : baseline == null ? null : summary[definition.key] - baseline,
      interval: normalizedInterval(intervals[definition.key]),
      improvementInterval,
      significanceHint: typeof paired?.significance_hint === "string" ? paired.significance_hint : null,
      kpiId: typeof assessment?.kpi_id === "string" ? assessment.kpi_id : null,
      status: typeof assessment?.status === "string" ? assessment.status : "not_evaluable",
      target: assessment?.target && typeof assessment.target === "object" ? assessment.target : {},
      reasons: Array.isArray(assessment?.reasons) ? assessment.reasons.map((reason) => ({
        code: String(reason?.code || "UNKNOWN"),
        message: String(reason?.message || ""),
      })) : [],
    };
  });
  const failureCases = (report?.failure_cases || []).map(normalizeFailureCase);
  const runEvidence = normalizeRunEvidence(bundle, report);
  const declaredPilotStatus = runEvidence.qualityStatus === "pilot_failed" ? "pilot_failed"
    : ["pilot_candidate", "pilot_passed"].includes(runEvidence.qualityStatus) ? "pilot_candidate" : null;
  const pilotDisposition = pilotOnly
    ? declaredPilotStatus || (comparison.every(absolutePilotPass) ? "pilot_candidate" : "pilot_failed")
    : null;
  return {
    sourceName,
    kind,
    schemaVersion: report?.schema_version || manifest?.schema_version || "1.0.0",
    truthState,
    verified,
    fixtureOnly,
    mockOnly,
    pilotOnly,
    pilotDisposition,
    fixtureNotice: manifest?.fixture_notice || metrics?.fixture_notice || null,
    watermark,
    eligibleForModelAcceptance: eligible,
    acceptanceStatus: String(acceptance.status || "not_evaluable"),
    createdAt: manifest?.created_at || null,
    datasetVersion: bundle.data_provenance?.dataset_version || null,
    dataManifestSha256: typeof bundle.data_provenance?.manifest_sha256 === "string"
      ? bundle.data_provenance.manifest_sha256 : null,
    formalKpiEligible: bundle.data_provenance?.formal_kpi_eligible ?? null,
    formalKpiIneligibilityReason: bundle.data_provenance?.formal_kpi_ineligibility_reason || null,
    frozenTest: bundle.data_provenance?.frozen_test ?? null,
    environment: bundle.environment || null,
    summary,
    comparison,
    slices: normalizeSlices(report, bundle.failures),
    failureCases,
    failureTaxonomy: bundle.failures?.by_code && typeof bundle.failures.by_code === "object" ? bundle.failures.by_code : {},
    sampleCount: summary.sample_count,
    comparisonStrength: pairedComparison?.conclusion_strength || null,
    comparisonNotice: pairedComparison?.notice || null,
    sampleIds: (report?.samples || []).map((sample) => sample.sample_id).sort(),
    ...runEvidence,
  };
}

export function compareEvaluationPortfolios(baseline, candidate) {
  const reasons = [];
  if (!baseline || !candidate) {
    reasons.push({ code: "PAIR_INCOMPLETE", message: "请分别导入零样本和 LoRA 的完整评测包。" });
  } else {
    for (const [role, portfolio] of [["ZERO_SHOT", baseline], ["CANDIDATE", candidate]]) {
      if (!portfolio.verified) reasons.push({ code: `${role}_PACKAGE_UNVERIFIED`, message: `${role} package 未通过组件 SHA-256。` });
      if (portfolio.fixtureOnly || portfolio.mockOnly) reasons.push({ code: `${role}_NOT_REAL`, message: `${role} package 是 fixture/mock。` });
      if (!portfolio.pilotOnly) reasons.push({ code: `${role}_NOT_PILOT`, message: `${role} package 未声明 pilot_only。` });
      if (!portfolio.dataManifestSha256) reasons.push({ code: `${role}_MANIFEST_MISSING`, message: `${role} package 未记录数据 manifest SHA-256。` });
      if (!portfolio.sampleIds.length) reasons.push({ code: `${role}_SAMPLE_IDS_MISSING`, message: `${role} report 未记录逐样本 ID。` });
      if (!portfolio.modelRevision) reasons.push({ code: `${role}_REVISION_MISSING`, message: `${role} package 未记录基础模型 revision。` });
    }
    if (baseline.dataManifestSha256 && candidate.dataManifestSha256
      && baseline.dataManifestSha256 !== candidate.dataManifestSha256) {
      reasons.push({ code: "DATA_MANIFEST_MISMATCH", message: "零样本与 LoRA 使用的数据 manifest 不同，已阻止对比。" });
    }
    if (baseline.sampleIds.length && candidate.sampleIds.length) {
      const left = baseline.sampleIds.join("\n");
      const right = candidate.sampleIds.join("\n");
      if (left !== right) reasons.push({ code: "SAMPLE_ID_MISMATCH", message: "零样本与 LoRA 的 sample ID 集合不一致，已阻止对比。" });
    }
    const candidateUsesVlmBase = candidate.candidateKind !== "specialist";
    if (candidateUsesVlmBase && !baseline.promptIdentity) {
      reasons.push({ code: "ZERO_SHOT_PROMPT_MISSING", message: "零样本 package 未记录 prompt 身份。" });
    }
    if (candidateUsesVlmBase && !candidate.promptIdentity) {
      reasons.push({ code: "CANDIDATE_PROMPT_MISSING", message: "LoRA/fused candidate 未记录 prompt 身份。" });
    } else if (candidateUsesVlmBase && baseline.promptIdentity && candidate.promptIdentity
      && baseline.promptIdentity !== candidate.promptIdentity) {
      reasons.push({ code: "PROMPT_MISMATCH", message: "零样本与 LoRA/fused candidate 的 prompt 不一致。" });
    }
    if (candidateUsesVlmBase && baseline.modelRevision && candidate.modelRevision
      && baseline.modelRevision !== candidate.modelRevision) {
      reasons.push({ code: "MODEL_REVISION_MISMATCH", message: "零样本与 LoRA/fused candidate 的基础模型 revision 不一致。" });
    }
    if (baseline.adapterHash) reasons.push({ code: "ZERO_SHOT_ADAPTER_PRESENT", message: "零样本包不应包含 adapter hash。" });
    if (candidate.candidateKind === "specialist") {
      if (!candidate.specialistIdentity) reasons.push({ code: "SPECIALIST_IDENTITY_MISSING", message: "specialist package 未记录模型身份。" });
    } else if (!candidate.adapterHash) {
      reasons.push({ code: "CANDIDATE_IDENTITY_MISSING", message: "LoRA/fused candidate 未记录 adapter hash。" });
    }
  }
  const allowed = reasons.length === 0;
  const metrics = allowed ? evaluationMetricDefinitions.map((definition) => {
    const left = baseline.comparison.find((metric) => metric.key === definition.key);
    const right = candidate.comparison.find((metric) => metric.key === definition.key);
    return {
      ...definition,
      baseline: left.candidate,
      candidate: right.candidate,
      delta: right.candidate - left.candidate,
      baselineInterval: left.interval,
      candidateInterval: right.interval,
    };
  }) : [];
  const qualityStatus = !baseline && !candidate ? "not_loaded"
    : !allowed ? "comparison_blocked"
    : metrics.every((metric) => absolutePilotPass({ key: metric.key, candidate: metric.candidate }))
      ? "pilot_candidate" : "pilot_failed";
  return {
    allowed,
    reasons,
    qualityStatus,
    qualityAccepted: false,
    formalKpiClaimAllowed: false,
    pilotSampleCount: allowed ? candidate.sampleCount : null,
    metrics,
  };
}

export function parseEvaluationPayload(payload, { filename = "evaluation.json", verified = false } = {}) {
  const value = objectValue(payload, filename);
  if (value.package_manifest || value.metrics || value.report) {
    return buildPortfolio(value, { sourceName: filename, verified, kind: "package_bundle" });
  }
  if (REQUIRED_REPORT_KEYS.every((key) => key in value)) {
    return buildPortfolio({ report: value }, { sourceName: filename, verified: false, kind: "report" });
  }
  if (value.summary && (value.confidence_intervals || value.kpi_acceptance || "fixture_only" in value)) {
    return buildPortfolio({ metrics: value }, { sourceName: filename, verified: false, kind: "metrics" });
  }
  throw new EvaluationImportError("UNRECOGNIZED_EVALUATION_JSON", "未识别的 evaluator report/package JSON。");
}

async function sha256Hex(buffer) {
  if (!globalThis.crypto?.subtle) {
    throw new EvaluationImportError("HASH_UNAVAILABLE", "当前环境无法校验 package SHA-256。");
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function parseEvaluationFiles(fileList) {
  const files = [...(fileList || [])];
  if (!files.length) throw new EvaluationImportError("NO_FILES", "请选择 evaluator JSON。");
  if (files.length > MAX_PACKAGE_FILES) {
    throw new EvaluationImportError("TOO_MANY_FILES", `一次最多导入 ${MAX_PACKAGE_FILES} 个 JSON 文件。`);
  }
  const documents = new Map();
  for (const file of files) {
    if (!file || typeof file.text !== "function" || typeof file.arrayBuffer !== "function") {
      throw new EvaluationImportError("INVALID_FILE", "导入项不是可读文件。");
    }
    if (!file.size || file.size > MAX_FILE_BYTES) {
      throw new EvaluationImportError("FILE_SIZE", `${file.name || "JSON"} 必须大于 0 且不超过 5 MB。`);
    }
    const name = String(file.name || "evaluation.json").split(/[\\/]/).pop();
    if (!name.toLowerCase().endsWith(".json")) {
      throw new EvaluationImportError("INVALID_FILE_TYPE", `${name} 不是 .json 文件。`);
    }
    if (documents.has(name)) throw new EvaluationImportError("DUPLICATE_FILE", `重复文件名：${name}。`);
    const bytes = await file.arrayBuffer();
    let value;
    try {
      value = JSON.parse(new TextDecoder().decode(bytes));
    } catch {
      throw new EvaluationImportError("INVALID_JSON", `${name} 不是有效 JSON。`);
    }
    documents.set(name, { value, bytes });
  }
  if (documents.size === 1) {
    const [name, document] = documents.entries().next().value;
    return parseEvaluationPayload(document.value, { filename: name });
  }
  const manifestDocument = documents.get("package_manifest.json");
  let verified = false;
  if (manifestDocument) {
    const manifest = objectValue(manifestDocument.value, "package_manifest.json");
    const hashes = objectValue(manifest.component_sha256, "component_sha256");
    for (const [name, expected] of Object.entries(hashes).filter(([name]) => name.toLowerCase().endsWith(".json"))) {
      const document = documents.get(name);
      if (!document) {
        throw new EvaluationImportError("PACKAGE_COMPONENT_MISSING", `package 缺少组件：${name}。请一次选中整个评测包的 JSON。`);
      }
      const actual = await sha256Hex(document.bytes);
      if (actual !== expected) {
        throw new EvaluationImportError("PACKAGE_HASH_MISMATCH", `${name} 的 SHA-256 与 package manifest 不一致。`);
      }
    }
    verified = true;
  }
  const bundle = {
    package_manifest: manifestDocument?.value,
    metrics: documents.get("metrics.json")?.value,
    report: documents.get("report.json")?.value,
    failures: documents.get("failures.json")?.value,
    config: documents.get("config.json")?.value,
    data_provenance: documents.get("data_provenance.json")?.value,
    environment: documents.get("environment.json")?.value,
    comparison: documents.get("comparison.json")?.value,
    run_manifest: documents.get("run_manifest.json")?.value,
    experiment: documents.get("experiment.json")?.value,
    performance: documents.get("performance.json")?.value,
  };
  return buildPortfolio(bundle, {
    sourceName: manifestDocument ? "evaluation-package" : "evaluation-components",
    verified,
    kind: manifestDocument ? "package" : "package_components",
  });
}

export const evaluationImportLimits = Object.freeze({
  maxFileBytes: MAX_FILE_BYTES,
  maxPackageFiles: MAX_PACKAGE_FILES,
});
