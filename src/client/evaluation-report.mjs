const MAX_FILE_BYTES = 5 * 1024 * 1024;
const MAX_PACKAGE_FILES = 12;

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
  const acceptance = metrics?.kpi_acceptance && typeof metrics.kpi_acceptance === "object"
    ? metrics.kpi_acceptance : { status: "not_evaluable", eligible_for_model_acceptance: false, assessments: [] };
  const eligible = verified && !fixtureOnly && !mockOnly && acceptance.eligible_for_model_acceptance === true;
  const truthState = fixtureOnly ? "fixture" : mockOnly ? "mock" : verified ? "verified" : "unverified";
  const watermark = fixtureOnly
    ? "FIXTURE / MOCK · 不可用于模型验收"
    : mockOnly ? "MOCK OUTPUT · 不可用于模型验收"
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
  return {
    sourceName,
    kind,
    schemaVersion: report?.schema_version || manifest?.schema_version || "1.0.0",
    truthState,
    verified,
    fixtureOnly,
    mockOnly,
    fixtureNotice: manifest?.fixture_notice || metrics?.fixture_notice || null,
    watermark,
    eligibleForModelAcceptance: eligible,
    acceptanceStatus: String(acceptance.status || "not_evaluable"),
    createdAt: manifest?.created_at || null,
    datasetVersion: bundle.data_provenance?.dataset_version || null,
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
