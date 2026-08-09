import {
  compareEvaluationPortfolios,
  EvaluationImportError,
  parseEvaluationFiles,
  parseEvaluationPayload,
} from "/src/client/evaluation-report.mjs";

const ids = [
  "toggle-evaluation", "evaluation-panel", "close-evaluation", "refresh-diagnostics",
  "runtime-diagnostics", "evaluation-input", "select-evaluation", "load-evaluation-fixture",
  "clear-evaluation", "evaluation-alert", "evaluation-empty", "evaluation-content",
  "evaluation-watermark", "evaluation-source", "evaluation-provenance-detail",
  "evaluation-truth-badge", "evaluation-acceptance", "evaluation-summary",
  "evaluation-comparison", "evaluation-kpis", "evaluation-slices", "evaluation-failures",
  "phase6-runtime-state", "phase6-quality-state", "phase6-baseline-slot", "phase6-candidate-slot",
  "phase6-baseline-input", "phase6-candidate-input", "select-phase6-baseline", "select-phase6-candidate",
  "phase6-baseline-detail", "phase6-candidate-detail", "phase6-compare-alert", "phase6-comparison-content",
  "phase6-identity-grid", "phase6-metrics", "phase6-resources", "phase6-slices",
];

function elementMap() {
  return Object.fromEntries(ids.map((id) => [id.replaceAll("-", "_"), document.getElementById(id)]));
}

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}

function formatPercent(value, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

function formatDelta(value) {
  if (typeof value !== "number") return "—";
  const points = value * 100;
  return `${points > 0 ? "+" : ""}${points.toFixed(1)} pp`;
}

function statusLabel(status) {
  return ({
    passed: "通过",
    failed: "未通过",
    not_evaluable: "不可评估",
    fixture_only: "FIXTURE",
    mock_only: "MOCK",
    pilot_only: "PILOT ONLY",
    pilot_failed: "PILOT FAILED",
    pilot_candidate: "PILOT CANDIDATE",
  })[status] || String(status || "unknown");
}

function compactText(value, fallback = "not recorded") {
  if (typeof value !== "string" || !value.trim()) return fallback;
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized.length > 92 ? `${normalized.slice(0, 89)}…` : normalized;
}

function intervalText(interval) {
  return interval?.lower != null && interval?.upper != null
    ? `${formatPercent(interval.lower)}–${formatPercent(interval.upper)} · N=${interval.observationCount ?? "—"}`
    : "not reported";
}

function resourceText(value, unit) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(value < 100 ? 1 : 0)} ${unit}` : "not recorded";
}

function identityCard(label, value, title = value) {
  const card = node("article", "phase6-identity-card");
  const strong = node("strong", "", compactText(value));
  strong.title = title || "not recorded";
  card.append(node("small", "", label), strong);
  return card;
}

function resourceCard(label, value) {
  const card = node("article", "phase6-resource-card");
  card.append(node("small", "", label), node("strong", "", value));
  return card;
}

function renderPhase6Slices(elements, baseline, candidate) {
  const column = (label, portfolio) => {
    const section = node("section", "phase6-slice-column");
    section.append(node("strong", "", label));
    const list = node("ul");
    const slices = portfolio.slices.slice(0, 5);
    if (!slices.length) list.append(node("li", "", "no slice metrics"));
    for (const slice of slices) {
      const item = node("li");
      item.append(node("span", "", slice.name), node("span", "", `fail ${formatPercent(slice.failureRate)} · N=${slice.sample_count}`));
      list.append(item);
    }
    section.append(list);
    return section;
  };
  elements.phase6_slices.replaceChildren(column("A / ZERO-SHOT FAILURE SLICES", baseline), column("B / LORA FAILURE SLICES", candidate));
}

function renderPhase6Pair(elements, pair, readiness) {
  const runtimeReady = readiness?.ready === true
    && readiness?.runtime?.selected_mode === "real"
    && readiness?.runtime?.degraded === false;
  elements.phase6_runtime_state.dataset.state = runtimeReady ? "runtime_ready" : "runtime_blocked";
  elements.phase6_runtime_state.textContent = runtimeReady ? "RUNTIME_READY" : "RUNTIME_BLOCKED";

  const comparison = compareEvaluationPortfolios(pair.baseline, pair.candidate);
  elements.phase6_quality_state.dataset.state = comparison.qualityStatus;
  elements.phase6_quality_state.textContent = comparison.qualityStatus.toUpperCase();
  if (!comparison.allowed) {
    elements.phase6_compare_alert.dataset.state = pair.baseline || pair.candidate ? "blocked" : "pending";
    elements.phase6_compare_alert.textContent = comparison.reasons.map((reason) => `${reason.code} · ${reason.message}`).join(" ");
    elements.phase6_comparison_content.hidden = true;
    elements.phase6_identity_grid.replaceChildren();
    elements.phase6_metrics.replaceChildren();
    elements.phase6_resources.replaceChildren();
    elements.phase6_slices.replaceChildren();
    return comparison;
  }

  const { baseline, candidate } = pair;
  elements.phase6_compare_alert.dataset.state = "pass";
  elements.phase6_compare_alert.textContent = `PAIR VERIFIED · ${comparison.pilotSampleCount} 个 sample ID 完全一致；quality_accepted=false，formal KPI claim locked。`;
  elements.phase6_comparison_content.hidden = false;
  elements.phase6_identity_grid.replaceChildren(
    identityCard("DATA MANIFEST SHA-256", baseline.dataManifestSha256),
    identityCard("PAIRED SAMPLE IDS", `${baseline.sampleIds.length} exact matches`),
    identityCard("PROMPT IDENTITY", baseline.promptDisplay, baseline.promptIdentity),
    identityCard("ZERO-SHOT REVISION", baseline.modelRevision),
    identityCard("LORA BASE REVISION", candidate.modelRevision),
    identityCard("LORA ADAPTER HASH", candidate.adapterHash),
    identityCard("ZERO-SHOT RUN", baseline.runId),
    identityCard("LORA RUN", candidate.runId),
  );

  const table = node("table", "comparison-table phase6-metric-table");
  const caption = node("caption", "sr-only", "Phase 6 零样本与 LoRA Pilot 指标对比");
  const head = node("thead");
  const header = node("tr");
  for (const label of ["指标", "零样本", "零样本 CI", "LoRA", "LoRA CI", "变化", "状态"]) header.append(node("th", "", label));
  head.append(header);
  const body = node("tbody");
  for (const metric of comparison.metrics) {
    const row = node("tr");
    const favorable = metric.direction === "lower" ? metric.delta <= 0 : metric.delta >= 0;
    row.append(
      node("th", "", metric.label),
      node("td", "metric-number", formatPercent(metric.baseline)),
      node("td", "metric-interval", intervalText(metric.baselineInterval)),
      node("td", "metric-number is-candidate", formatPercent(metric.candidate)),
      node("td", "metric-interval", intervalText(metric.candidateInterval)),
      node("td", "metric-number", formatDelta(metric.delta)),
    );
    const statusCell = node("td");
    const badge = node("span", "metric-status", favorable ? "direction OK" : "direction REGRESSED");
    badge.dataset.status = favorable ? "pilot_only" : "failed";
    statusCell.append(badge);
    row.append(statusCell);
    body.append(row);
  }
  table.append(caption, head, body);
  elements.phase6_metrics.replaceChildren(table);
  elements.phase6_resources.replaceChildren(
    resourceCard("ZERO P50", resourceText(baseline.resources.p50LatencyMs, "ms")),
    resourceCard("ZERO P95", resourceText(baseline.resources.p95LatencyMs, "ms")),
    resourceCard("ZERO RSS PEAK", resourceText(baseline.resources.processPeakRssMb ?? baseline.resources.peakMemoryMb, "MB")),
    resourceCard("ZERO MLX PEAK", resourceText(baseline.resources.mlxPeakAllocatedMb, "MB")),
    resourceCard("LORA P50", resourceText(candidate.resources.p50LatencyMs, "ms")),
    resourceCard("LORA P95", resourceText(candidate.resources.p95LatencyMs, "ms")),
    resourceCard("LORA RSS PEAK", resourceText(candidate.resources.processPeakRssMb ?? candidate.resources.peakMemoryMb, "MB")),
    resourceCard("LORA MLX PEAK", resourceText(candidate.resources.mlxPeakAllocatedMb, "MB")),
  );
  renderPhase6Slices(elements, baseline, candidate);
  return comparison;
}

function renderDiagnostics(elements, readiness) {
  const runtime = readiness?.runtime || {};
  const models = Array.isArray(readiness?.models) ? readiness.models : [];
  const aliasText = Object.entries(readiness?.aliases || {}).map(([alias, model]) => `${alias}→${model}`).join(" · ") || "no aliases";
  const overview = node("article", "diagnostic-overview");
  const state = node("span", "diagnostic-state", readiness?.label || "未获取 readiness");
  state.dataset.state = readiness?.state || "unavailable";
  overview.append(
    state,
    node("strong", "", `selected · ${runtime.selected_mode || "unknown"}`),
    node("small", "", `requested ${runtime.requested_mode || "unknown"} · request_id ${readiness?.requestId || "—"}`),
    node("small", "", `aliases · ${aliasText}`),
  );
  const reason = node("p", "diagnostic-reason");
  reason.textContent = runtime.degraded
    ? `降级原因 · ${runtime.fallback_reason || readiness?.detail || "未提供"}`
    : `运行诊断 · ${readiness?.detail || "无降级"}`;
  overview.append(reason);

  const modelList = node("div", "lifecycle-list");
  if (!models.length) {
    modelList.append(node("p", "diagnostic-empty", readiness?.state === "mock"
      ? "离线浏览器 Mock 没有后端模型注册表。"
      : "readiness 未返回模型生命周期记录。"));
  } else {
    for (const model of models) {
      const card = node("article", "lifecycle-card");
      card.dataset.state = model.state || "unknown";
      const head = node("div", "lifecycle-head");
      head.append(node("strong", "", model.model_id || "unknown"));
      const badge = node("span", "lifecycle-state", String(model.state || "unknown"));
      badge.dataset.state = model.state || "unknown";
      head.append(badge);
      const ready = node("span", "model-ready", model.ready ? "READY" : "NOT READY");
      ready.dataset.ready = String(Boolean(model.ready));
      card.append(
        head,
        node("p", "", `${model.base || "unknown base"} · ${model.adapter || "no adapter"}`),
        node("small", "", `${model.source || "unknown source"} · ${model.quantization || "quantization n/a"}`),
        node("small", "", `weight ${model.weight_hash ? String(model.weight_hash).slice(0, 16) : "not recorded"}`),
        ready,
      );
      modelList.append(card);
    }
  }
  elements.runtime_diagnostics.replaceChildren(overview, modelList);
}

function renderComparison(elements, portfolio) {
  const table = node("table", "comparison-table");
  const caption = node("caption", "sr-only", "零样本基线与候选模型指标对比");
  const header = node("tr");
  for (const label of ["KPI / 指标", "零样本", "候选", "变化", "置信区间", "状态"]) header.append(node("th", "", label));
  const head = node("thead");
  head.append(header);
  const body = node("tbody");
  for (const metric of portfolio.comparison) {
    const row = node("tr");
    row.dataset.status = metric.status;
    const title = node("th");
    title.scope = "row";
    title.append(node("strong", "", metric.label), node("small", "", metric.kpiId || metric.key));
    const interval = metric.interval;
    let intervalText = interval && interval.lower != null && interval.upper != null
      ? `${formatPercent(interval.lower)} – ${formatPercent(interval.upper)}\nN=${interval.observationCount ?? "—"}`
      : "—";
    if (metric.improvementInterval?.lower != null && metric.improvementInterval?.upper != null) {
      intervalText += `\nΔ ${formatDelta(metric.improvementInterval.lower)} – ${formatDelta(metric.improvementInterval.upper)}`;
    }
    const status = node("span", "metric-status", statusLabel(metric.status));
    status.dataset.status = metric.status;
    row.append(
      title,
      node("td", "metric-number", formatPercent(metric.baseline)),
      node("td", "metric-number is-candidate", formatPercent(metric.candidate)),
      node("td", "metric-number", formatDelta(metric.delta)),
      node("td", "metric-interval", intervalText),
      node("td", "", ""),
    );
    row.lastElementChild.append(status);
    body.append(row);
  }
  table.append(caption, head, body);
  elements.evaluation_comparison.replaceChildren(table);
}

function renderKpis(elements, portfolio) {
  const cards = portfolio.comparison.map((metric) => {
    const card = node("article", "kpi-card");
    card.dataset.status = metric.status;
    const top = node("div", "kpi-head");
    top.append(node("strong", "", metric.kpiId || metric.key));
    const badge = node("span", "metric-status", statusLabel(metric.status));
    badge.dataset.status = metric.status;
    top.append(badge);
    card.append(top, node("p", "", metric.label));
    if (metric.reasons.length) {
      const list = node("ul", "kpi-reasons");
      for (const reason of metric.reasons) list.append(node("li", "", `${reason.code} · ${reason.message}`));
      card.append(list);
    } else {
      card.append(node("small", "", "无阻断原因"));
    }
    if (metric.significanceHint) card.append(node("small", "kpi-hint", `paired · ${metric.significanceHint}`));
    return card;
  });
  elements.evaluation_kpis.replaceChildren(...cards);
}

function renderSlices(elements, portfolio) {
  if (!portfolio.slices.length) {
    elements.evaluation_slices.replaceChildren(node("p", "lab-empty", "当前导入不含 slice_metrics。"));
    return;
  }
  const items = portfolio.slices.map((slice) => {
    const item = node("article", "slice-item");
    const top = node("div", "slice-head");
    top.append(node("strong", "", slice.name), node("span", "", `N=${slice.sample_count}`));
    const bar = node("span", "slice-bar");
    bar.style.setProperty("--failure-rate", `${Math.max(0, Math.min(1, slice.failureRate)) * 100}%`);
    item.append(
      top,
      bar,
      node("small", "", `失败 ${slice.failureCount} · ${formatPercent(slice.failureRate)} · F1 ${formatPercent(slice.macro_f1)} · IoU ${formatPercent(slice.acc_at_iou)}`),
    );
    return item;
  });
  elements.evaluation_slices.replaceChildren(...items);
}

function renderFailures(elements, portfolio) {
  if (!portfolio.failureCases.length) {
    elements.evaluation_failures.replaceChildren(node("p", "lab-empty", "报告中没有失败案例。"));
    return;
  }
  const items = portfolio.failureCases.slice(0, 50).map((failure) => {
    const item = node("article", "failure-item");
    const heading = node("div", "failure-head");
    heading.append(node("strong", "", failure.sampleId), node("span", "", failure.source));
    const codes = node("div", "failure-codes");
    for (const code of failure.codes) codes.append(node("code", "", code));
    item.append(
      heading,
      node("p", "", `truth ${failure.truth} → prediction ${failure.prediction}`),
      codes,
    );
    return item;
  });
  if (portfolio.failureCases.length > 50) items.push(node("p", "lab-empty", `仅显示前 50 / ${portfolio.failureCases.length} 个失败案例。`));
  elements.evaluation_failures.replaceChildren(...items);
}

function renderPortfolio(elements, portfolio) {
  elements.evaluation_empty.hidden = true;
  elements.evaluation_content.hidden = false;
  elements.evaluation_content.dataset.truthState = portfolio.truthState;
  elements.evaluation_watermark.hidden = !portfolio.watermark;
  elements.evaluation_watermark.textContent = portfolio.watermark || "";
  elements.evaluation_source.textContent = portfolio.sourceName;
  elements.evaluation_provenance_detail.textContent = [
    portfolio.datasetVersion ? `dataset ${portfolio.datasetVersion}` : null,
    portfolio.createdAt ? `created ${portfolio.createdAt}` : null,
    portfolio.verified ? "SHA-256 verified" : "provenance unverified",
    portfolio.frozenTest === true ? "frozen test" : null,
  ].filter(Boolean).join(" · ");
  elements.evaluation_truth_badge.dataset.state = portfolio.truthState;
  elements.evaluation_truth_badge.textContent = ({ fixture: "FIXTURE", mock: "MOCK OUTPUT", pilot: "VERIFIED PILOT", verified: "PACKAGE VERIFIED", unverified: "UNVERIFIED" })[portfolio.truthState];
  elements.evaluation_acceptance.dataset.state = portfolio.acceptanceStatus;
  elements.evaluation_acceptance.textContent = portfolio.pilotDisposition
    ? `${portfolio.pilotDisposition.toUpperCase()} · FORMAL KPI LOCKED`
    : portfolio.eligibleForModelAcceptance
    ? "MODEL ACCEPTANCE ELIGIBLE"
    : `${statusLabel(portfolio.acceptanceStatus).toUpperCase()} · NOT ELIGIBLE`;
  elements.evaluation_summary.replaceChildren(
    node("span", "", `N ${portfolio.summary.sample_count}`),
    node("span", "", `FAIL ${portfolio.summary.failure_case_count}`),
    node("span", "", `HARD-NEG ${portfolio.summary.hard_negative_count}`),
    ...(portfolio.comparisonStrength ? [node("span", "", `PAIR ${portfolio.comparisonStrength}`)] : []),
  );
  renderComparison(elements, portfolio);
  renderKpis(elements, portfolio);
  renderSlices(elements, portfolio);
  renderFailures(elements, portfolio);
}

export function initializeEvaluationPanel({ getReadiness, refreshReadiness, onPortfolioChange = () => {} }) {
  const elements = elementMap();
  let importSequence = 0;
  const phase6Pair = { baseline: null, candidate: null };
  let phase6ImportSequence = 0;

  const showAlert = (message = "", code = "") => {
    elements.evaluation_alert.textContent = message ? `${code ? `${code} · ` : ""}${message}` : "";
    elements.evaluation_alert.hidden = !message;
  };
  const resetPortfolio = (message = "", invalidate = true) => {
    if (invalidate) importSequence += 1;
    elements.evaluation_content.hidden = true;
    elements.evaluation_empty.hidden = false;
    if (message) elements.evaluation_empty.querySelector("p").textContent = message;
    elements.evaluation_watermark.hidden = true;
    onPortfolioChange(null);
  };
  const setBusy = (busy) => {
    elements.evaluation_panel.setAttribute("aria-busy", String(busy));
    elements.select_evaluation.disabled = busy;
    elements.load_evaluation_fixture.disabled = busy;
  };
  const renderPhase6Slot = (role, portfolio, state = "verified") => {
    const slot = role === "baseline" ? elements.phase6_baseline_slot : elements.phase6_candidate_slot;
    const detail = role === "baseline" ? elements.phase6_baseline_detail : elements.phase6_candidate_detail;
    slot.dataset.state = state;
    detail.textContent = portfolio
      ? `${portfolio.sourceName} · N=${portfolio.sampleCount} · ${portfolio.pilotDisposition || portfolio.acceptanceStatus} · ${portfolio.verified ? "SHA verified" : "unverified"}`
      : role === "baseline" ? "等待 package_manifest 及全部 JSON 组件" : "等待 adapter hash 与评测组件";
  };
  const importPhase6 = async (role) => {
    const input = role === "baseline" ? elements.phase6_baseline_input : elements.phase6_candidate_input;
    const button = role === "baseline" ? elements.select_phase6_baseline : elements.select_phase6_candidate;
    const token = ++phase6ImportSequence;
    phase6Pair[role] = null;
    renderPhase6Slot(role, null, "empty");
    renderPhase6Pair(elements, phase6Pair, getReadiness());
    button.disabled = true;
    try {
      const portfolio = await parseEvaluationFiles(input.files);
      if (token !== phase6ImportSequence) return;
      phase6Pair[role] = portfolio;
      renderPhase6Slot(role, portfolio, portfolio.verified ? "verified" : "error");
      const comparison = renderPhase6Pair(elements, phase6Pair, getReadiness());
      if (role === "candidate") {
        renderPortfolio(elements, portfolio);
        onPortfolioChange(portfolio);
      }
      if (!comparison.allowed && phase6Pair.baseline && phase6Pair.candidate) onPortfolioChange(null);
    } catch (error) {
      if (token !== phase6ImportSequence) return;
      renderPhase6Slot(role, null, "error");
      const known = error instanceof EvaluationImportError;
      elements.phase6_compare_alert.dataset.state = "blocked";
      elements.phase6_compare_alert.textContent = `${known ? error.code : "IMPORT_FAILED"} · ${error.message || "无法读取 Phase 6 package。"}`;
      elements.phase6_comparison_content.hidden = true;
      onPortfolioChange(null);
    } finally {
      if (token === phase6ImportSequence) button.disabled = false;
      input.value = "";
    }
  };
  const acceptPortfolio = (portfolio, token) => {
    if (token !== importSequence) return;
    renderPortfolio(elements, portfolio);
    onPortfolioChange(portfolio);
    showAlert();
  };
  const failImport = (error, token) => {
    if (token !== importSequence) return;
    resetPortfolio("导入失败；已清空上一份评测证据，避免陈旧指标混入。", false);
    const known = error instanceof EvaluationImportError;
    showAlert(error.message || "无法读取评测 JSON。", known ? error.code : "IMPORT_FAILED");
  };
  const importFiles = async () => {
    resetPortfolio("正在校验 evaluator 证据…");
    const token = importSequence;
    setBusy(true);
    try {
      acceptPortfolio(await parseEvaluationFiles(elements.evaluation_input.files), token);
    } catch (error) {
      failImport(error, token);
    } finally {
      if (token === importSequence) setBusy(false);
      elements.evaluation_input.value = "";
    }
  };
  const loadFixture = async () => {
    resetPortfolio("正在加载 fixture 契约样例…");
    const token = importSequence;
    setBusy(true);
    try {
      const response = await fetch("/ui/fixtures/evaluation-package.fixture.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      acceptPortfolio(parseEvaluationPayload(payload, { filename: "evaluation-package.fixture.json" }), token);
    } catch (error) {
      failImport(error, token);
    } finally {
      if (token === importSequence) setBusy(false);
    }
  };
  const openPanel = () => {
    elements.evaluation_panel.hidden = false;
    elements.toggle_evaluation.setAttribute("aria-expanded", "true");
    renderDiagnostics(elements, getReadiness());
    elements.evaluation_panel.focus({ preventScroll: true });
    elements.evaluation_panel.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const closePanel = () => {
    elements.evaluation_panel.hidden = true;
    elements.toggle_evaluation.setAttribute("aria-expanded", "false");
    elements.toggle_evaluation.focus();
  };

  elements.toggle_evaluation.addEventListener("click", () => elements.evaluation_panel.hidden ? openPanel() : closePanel());
  elements.close_evaluation.addEventListener("click", closePanel);
  elements.select_evaluation.addEventListener("click", () => elements.evaluation_input.click());
  elements.evaluation_input.addEventListener("change", importFiles);
  elements.load_evaluation_fixture.addEventListener("click", loadFixture);
  elements.select_phase6_baseline.addEventListener("click", () => elements.phase6_baseline_input.click());
  elements.select_phase6_candidate.addEventListener("click", () => elements.phase6_candidate_input.click());
  elements.phase6_baseline_input.addEventListener("change", () => importPhase6("baseline"));
  elements.phase6_candidate_input.addEventListener("change", () => importPhase6("candidate"));
  elements.clear_evaluation.addEventListener("click", () => { resetPortfolio("评测证据已清空。"); showAlert(); });
  elements.refresh_diagnostics.addEventListener("click", async () => {
    elements.refresh_diagnostics.disabled = true;
    try {
      await refreshReadiness();
      renderDiagnostics(elements, getReadiness());
      showAlert();
    } catch (error) {
      renderDiagnostics(elements, getReadiness());
      showAlert(error.message || "readiness 刷新失败。", error.code || "MODEL_NOT_READY");
    } finally {
      elements.refresh_diagnostics.disabled = false;
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !elements.evaluation_panel.hidden) closePanel();
  });
  renderDiagnostics(elements, getReadiness());
  renderPhase6Pair(elements, phase6Pair, getReadiness());
  return {
    updateReadiness(readiness) {
      if (!elements.evaluation_panel.hidden) renderDiagnostics(elements, readiness);
      renderPhase6Pair(elements, phase6Pair, readiness);
    },
  };
}
