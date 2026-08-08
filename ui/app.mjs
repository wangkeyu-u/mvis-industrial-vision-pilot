import { AnalysisClientError, createAnalysisClient, validateImageFile } from "/src/client/api-client.mjs";
import { initializeEvaluationPanel } from "/ui/evaluation-panel.mjs";

const params = new URLSearchParams(window.location.search);
const clientMode = params.get("client") === "api" ? "api" : "mock";
const endpoint = params.get("endpoint") || "/v1/analyze";
const transport = params.get("transport") === "json" ? "json" : "multipart";
const configuredTimeout = Number(params.get("timeout_ms"));
const requestTimeoutMs = Number.isFinite(configuredTimeout) && configuredTimeout > 0
  ? Math.min(configuredTimeout, 120000)
  : 10000;
const client = createAnalysisClient({ mode: clientMode, endpoint, transport, requestTimeoutMs });

const elements = Object.fromEntries([
  "image-input", "drop-zone", "demo-sample", "file-summary", "file-name", "file-meta", "query-input",
  "task-select", "model-select", "specialist-toggle", "form-alert", "analyze-button",
  "preview-image", "overlay-canvas", "image-stage", "stage-empty", "stage-loader",
  "case-state", "result-card", "result-icon", "result-label", "result-reason", "warning-list",
  "evidence-list",
  "model-base", "model-adapter", "latency-total", "latency-detail", "trace-source",
  "request-id", "copy-json", "download-json", "download-evidence", "clear-case", "client-mode",
  "runtime-status", "runtime-label",
].map((id) => [id.replaceAll("-", "_"), document.getElementById(id)]));

let selectedFile = null;
let imageDimensions = null;
let latestResult = null;
let latestError = null;
let latestReadiness = null;
let previewUrl = null;
let controller = null;
let isAnalyzing = false;
let requestSequence = 0;
let evaluationPanel = null;

const statePresentation = {
  idle: ["—", "尚未运行审查", "等待输入"],
  violation: ["!", "发现疑似违规", "需复核"],
  compliant: ["✓", "未发现明确违规", "通过"],
  uncertain: ["?", "证据不足", "不确定"],
  refused: ["×", "系统拒绝回答", "已拒答"],
  error: ["!", "请求未完成", "错误"],
};

function setRuntimeStatus(readiness) {
  latestReadiness = readiness;
  elements.runtime_status.dataset.state = readiness.state;
  elements.runtime_label.textContent = readiness.label;
  elements.client_mode.textContent = clientMode === "api"
    ? `${transport} · ${endpoint} · ${readiness.detail}`
    : readiness.detail;
  evaluationPanel?.updateReadiness(readiness);
}

async function refreshReadiness({ signal } = {}) {
  if (clientMode === "api") {
    setRuntimeStatus({
      ready: false,
      state: "checking",
      label: "检查 FastAPI readiness…",
      detail: `${transport} · ${endpoint}`,
      requestId: null,
      runtime: {},
      activeModel: null,
    });
  }
  try {
    const readiness = await client.checkReadiness({ signal });
    setRuntimeStatus(readiness);
    return readiness;
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    const readiness = {
      ready: false,
      state: "unavailable",
      label: "FastAPI 不可用",
      detail: error.message || "readiness 预检失败",
      requestId: error.requestId || null,
      runtime: error.details?.runtime || {},
      activeModel: null,
    };
    setRuntimeStatus(readiness);
    throw error;
  }
}

function showAlert(message = "") {
  elements.form_alert.textContent = message;
  elements.form_alert.hidden = !message;
}

function setState(state, reason = "") {
  const [icon, label, badge] = statePresentation[state] || statePresentation.error;
  elements.result_card.dataset.state = state;
  elements.case_state.dataset.state = state;
  elements.result_icon.textContent = icon;
  elements.result_label.textContent = label;
  elements.case_state.textContent = badge;
  if (reason) elements.result_reason.textContent = reason;
}

function formatBytes(bytes) {
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function resetJsonExport() {
  elements.download_json.removeAttribute("href");
  elements.download_json.removeAttribute("download");
  elements.download_json.setAttribute("aria-disabled", "true");
  elements.copy_json.disabled = true;
}

function resetAcceptanceExport() {
  elements.download_evidence.removeAttribute("href");
  elements.download_evidence.removeAttribute("download");
  elements.download_evidence.setAttribute("aria-disabled", "true");
}

function prepareJsonExport(result) {
  resetJsonExport();
  const json = JSON.stringify(result, null, 2);
  elements.download_json.href = `data:application/json;charset=utf-8,${encodeURIComponent(json)}`;
  elements.download_json.download = `${result.request_id || "visual-review-result"}.json`;
  elements.download_json.setAttribute("aria-disabled", "false");
  elements.copy_json.disabled = false;
}

function prepareAcceptanceExport(outcome) {
  resetAcceptanceExport();
  const requestId = outcome.request_id || latestReadiness?.requestId || "no-request-id";
  const evidence = {
    artifact_type: "mvis_demo_acceptance_evidence",
    schema_version: "1.0",
    generated_at: new Date().toISOString(),
    client: {
      mode: clientMode,
      endpoint,
      transport: clientMode === "api" ? transport : "in-browser",
      timeout_ms: requestTimeoutMs,
    },
    readiness: latestReadiness ? {
      ready: latestReadiness.ready,
      state: latestReadiness.state,
      label: latestReadiness.label,
      request_id: latestReadiness.requestId,
      runtime: latestReadiness.runtime,
      active_model: latestReadiness.activeModel,
    } : null,
    input: {
      file_name: selectedFile?.name || null,
      mime_type: selectedFile?.type || null,
      byte_size: selectedFile?.size || null,
      width: imageDimensions?.width || null,
      height: imageDimensions?.height || null,
      query_characters: elements.query_input.value.trim().length,
      task: elements.task_select.value,
      model: elements.model_select.value,
      use_specialist: elements.specialist_toggle.checked,
      image_bytes_embedded: false,
      query_text_embedded: false,
    },
    outcome,
    result: latestResult,
    stale_result_protection: {
      displayed_evidence_count: elements.evidence_list.children.length,
      result_export_available: elements.download_json.hasAttribute("href"),
      overlay_object_count: latestResult?.objects?.length || 0,
    },
  };
  const json = JSON.stringify(evidence, null, 2);
  elements.download_evidence.href = `data:application/json;charset=utf-8,${encodeURIComponent(json)}`;
  elements.download_evidence.download = `${requestId}-acceptance-evidence.json`;
  elements.download_evidence.setAttribute("aria-disabled", "false");
}

function clearDisplayedOutcome(reason = "等待当前请求返回新证据。") {
  latestResult = null;
  latestError = null;
  resetJsonExport();
  resetAcceptanceExport();
  elements.warning_list.replaceChildren();
  elements.evidence_list.replaceChildren();
  elements.evidence_list.hidden = true;
  elements.model_base.textContent = "—";
  elements.model_adapter.textContent = "等待结果";
  elements.latency_total.textContent = "—";
  elements.latency_detail.textContent = "预处理 / 推理 / 校验";
  elements.trace_source.textContent = "—";
  elements.request_id.textContent = "等待 request_id";
  setState("idle", reason);
  drawBoxes([]);
}

async function selectFile(file) {
  showAlert();
  try {
    await validateImageFile(file);
  } catch (error) {
    showAlert(error.message);
    return;
  }
  requestSequence += 1;
  controller?.abort();
  isAnalyzing = false;
  elements.stage_loader.hidden = true;
  elements.analyze_button.classList.remove("is-cancel");
  elements.analyze_button.firstElementChild.textContent = "开始审查";
  elements.analyze_button.lastElementChild.textContent = "⌘ ↵";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  imageDimensions = null;
  previewUrl = URL.createObjectURL(file);
  elements.preview_image.src = previewUrl;
  elements.preview_image.hidden = false;
  elements.stage_empty.hidden = true;
  elements.image_stage.dataset.empty = "false";
  elements.file_name.textContent = file.name || "未命名图片";
  elements.file_meta.textContent = `${file.type} · ${formatBytes(file.size)}`;
  elements.file_summary.hidden = false;
  clearDisplayedOutcome("图片已就绪；提交后将按原图像素坐标绘制证据框。");
  elements.preview_image.onload = () => {
    imageDimensions = { width: elements.preview_image.naturalWidth, height: elements.preview_image.naturalHeight };
    elements.file_meta.textContent = `${imageDimensions.width} × ${imageDimensions.height} · ${formatBytes(file.size)}`;
    drawBoxes([]);
  };
}

async function loadDemoSample(event) {
  event.preventDefault();
  event.stopPropagation();
  showAlert();
  try {
    const response = await fetch("/assets/system_architecture.png", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    await selectFile(new File([blob], "system_architecture.png", { type: "image/png" }));
  } catch {
    showAlert("内置演示图加载失败，请手动选择图片。");
  }
}

function renderedImageRect() {
  const stage = elements.image_stage.getBoundingClientRect();
  const image = elements.preview_image.getBoundingClientRect();
  return { x: image.left - stage.left, y: image.top - stage.top, width: image.width, height: image.height };
}

function drawBoxes(objects = []) {
  const canvas = elements.overlay_canvas;
  const stage = elements.image_stage.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(stage.width * ratio));
  canvas.height = Math.max(1, Math.round(stage.height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, stage.width, stage.height);
  if (!imageDimensions || !objects.length) return;
  const rendered = renderedImageRect();
  const scaleX = rendered.width / imageDimensions.width;
  const scaleY = rendered.height / imageDimensions.height;
  context.font = '700 12px "Noto Sans CJK SC", sans-serif';
  context.lineWidth = 2;
  for (const object of objects) {
    const [x1, y1, x2, y2] = object.bbox;
    const x = rendered.x + x1 * scaleX;
    const y = rendered.y + y1 * scaleY;
    const width = (x2 - x1) * scaleX;
    const height = (y2 - y1) * scaleY;
    context.strokeStyle = "#ff4d2d";
    context.fillStyle = "rgba(255, 77, 45, .08)";
    context.fillRect(x, y, width, height);
    context.strokeRect(x, y, width, height);
    const label = `${object.label}  ${(object.confidence * 100).toFixed(0)}% · ${object.source}`;
    const labelWidth = Math.min(context.measureText(label).width + 16, rendered.width);
    const labelY = Math.max(rendered.y, y - 26);
    context.fillStyle = "#ff4d2d";
    context.fillRect(x, labelY, labelWidth, 24);
    context.fillStyle = "#fffdf5";
    context.fillText(label, x + 8, labelY + 16, labelWidth - 12);
    const coordinate = `[${x1}, ${y1}, ${x2}, ${y2}]`;
    context.fillStyle = "rgba(16, 40, 45, .88)";
    context.fillRect(x, y + height - 20, Math.min(context.measureText(coordinate).width + 12, width), 20);
    context.fillStyle = "#fffdf5";
    context.font = '10px ui-monospace, monospace';
    context.fillText(coordinate, x + 6, y + height - 6, Math.max(0, width - 10));
  }
}

function renderResult(result) {
  latestResult = result;
  latestError = null;
  setState(result.policy_state || result.result, result.reason);
  elements.warning_list.replaceChildren(...(result.warnings || []).map((warning) => {
    const item = document.createElement("p");
    item.textContent = warning;
    return item;
  }));
  const objects = result.objects || [];
  elements.evidence_list.replaceChildren(...objects.map((object) => {
    const item = document.createElement("article");
    item.className = "evidence-item";
    const label = document.createElement("strong");
    label.textContent = object.label;
    const confidence = document.createElement("span");
    confidence.className = "evidence-confidence";
    confidence.textContent = Number.isFinite(object.confidence) ? `${(object.confidence * 100).toFixed(0)}%` : "—";
    confidence.setAttribute("aria-label", `置信度 ${confidence.textContent}`);
    const bbox = document.createElement("code");
    bbox.textContent = `[${(object.bbox || []).join(", ")}]`;
    const source = document.createElement("small");
    source.textContent = `来源 · ${object.source || "unknown"}`;
    item.append(label, confidence, bbox, source);
    return item;
  }));
  elements.evidence_list.hidden = objects.length === 0;
  elements.model_base.textContent = result.model?.base || "未知模型";
  elements.model_adapter.textContent = `${result.model?.adapter || "无适配器"} · ${result.model?.quantization || "未标记"}`;
  elements.latency_total.textContent = `${result.latency_ms ?? "—"} ms`;
  const timing = result.timing || {};
  elements.latency_detail.textContent = `${timing.preprocess_ms ?? "—"} / ${timing.inference_ms ?? "—"} / ${timing.validation_ms ?? "—"} ms`;
  elements.trace_source.textContent = result.trace?.source || "unknown";
  elements.request_id.textContent = result.request_id || "无 request_id";
  prepareJsonExport(result);
  drawBoxes(objects);
  prepareAcceptanceExport({
    kind: "success",
    request_id: result.request_id,
    result: result.policy_state || result.result,
    uncertain: result.uncertain,
  });
}

function renderError(error) {
  latestResult = null;
  resetJsonExport();
  const code = error.name === "AbortError"
    ? "REQUEST_CANCELLED"
    : error instanceof AnalysisClientError ? error.code : "INTERNAL_ERROR";
  const message = error.name === "AbortError" ? "请求已取消。" : error.message || "发生未知错误。";
  latestError = {
    kind: error.name === "AbortError" ? "cancelled" : "error",
    code,
    message,
    status: error.status || 0,
    request_id: error.requestId || null,
    details: error.details || null,
  };
  setState("error", `${code} · ${message}`);
  elements.warning_list.replaceChildren();
  elements.evidence_list.replaceChildren();
  elements.evidence_list.hidden = true;
  elements.trace_source.textContent = "error";
  elements.request_id.textContent = error.requestId || "无 request_id";
  drawBoxes([]);
  prepareAcceptanceExport(latestError);
}

async function analyze() {
  if (isAnalyzing) {
    controller?.abort();
    elements.analyze_button.firstElementChild.textContent = "正在取消";
    return;
  }
  showAlert();
  if (!selectedFile) return showAlert("请先上传一张合法图片。"), elements.drop_zone.focus();
  if (!elements.query_input.value.trim()) return showAlert("请输入审查要求。"), elements.query_input.focus();
  clearDisplayedOutcome();
  controller?.abort();
  controller = new AbortController();
  const requestToken = ++requestSequence;
  isAnalyzing = true;
  elements.analyze_button.classList.add("is-cancel");
  elements.stage_loader.hidden = false;
  elements.analyze_button.firstElementChild.textContent = "取消请求";
  elements.analyze_button.lastElementChild.textContent = "ESC";
  try {
    if (clientMode === "api") await refreshReadiness({ signal: controller.signal });
    const result = await client.analyze({
      image: selectedFile,
      imageWidth: imageDimensions?.width,
      imageHeight: imageDimensions?.height,
      query: elements.query_input.value,
      task: elements.task_select.value,
      model: elements.model_select.value,
      useSpecialist: elements.specialist_toggle.checked,
    }, { signal: controller.signal });
    if (requestToken === requestSequence) renderResult(result);
  } catch (error) {
    if (requestToken === requestSequence) renderError(error);
  } finally {
    if (requestToken === requestSequence) {
      isAnalyzing = false;
      elements.stage_loader.hidden = true;
      elements.analyze_button.classList.remove("is-cancel");
      elements.analyze_button.firstElementChild.textContent = "开始审查";
      elements.analyze_button.lastElementChild.textContent = "⌘ ↵";
    }
  }
}

function clearCase() {
  requestSequence += 1;
  isAnalyzing = false;
  controller?.abort();
  elements.stage_loader.hidden = true;
  elements.analyze_button.classList.remove("is-cancel");
  elements.analyze_button.firstElementChild.textContent = "开始审查";
  elements.analyze_button.lastElementChild.textContent = "⌘ ↵";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = null;
  imageDimensions = null;
  previewUrl = null;
  elements.image_input.value = "";
  elements.preview_image.removeAttribute("src");
  elements.preview_image.hidden = true;
  elements.stage_empty.hidden = false;
  elements.file_summary.hidden = true;
  elements.image_stage.dataset.empty = "true";
  clearDisplayedOutcome("提交图片和审查要求后，系统将在这里展示结论、解释及风险提示。");
  showAlert();
  drawBoxes([]);
}

async function copyJson() {
  if (!latestResult) return;
  await navigator.clipboard.writeText(JSON.stringify(latestResult, null, 2));
  const original = elements.copy_json.textContent;
  elements.copy_json.textContent = "已复制";
  setTimeout(() => { elements.copy_json.textContent = original; }, 1200);
}

elements.drop_zone.addEventListener("click", () => elements.image_input.click());
elements.drop_zone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); elements.image_input.click(); }
});
elements.image_input.addEventListener("change", () => selectFile(elements.image_input.files[0]));
elements.demo_sample.addEventListener("click", loadDemoSample);
for (const eventName of ["dragenter", "dragover"]) {
  elements.drop_zone.addEventListener(eventName, (event) => { event.preventDefault(); elements.drop_zone.classList.add("is-dragging"); });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.drop_zone.addEventListener(eventName, (event) => { event.preventDefault(); elements.drop_zone.classList.remove("is-dragging"); });
}
elements.drop_zone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
document.querySelectorAll(".scenario-chip").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".scenario-chip").forEach((chip) => chip.classList.toggle("is-active", chip === button));
  elements.query_input.value = button.dataset.query;
}));
elements.analyze_button.addEventListener("click", analyze);
elements.clear_case.addEventListener("click", clearCase);
elements.copy_json.addEventListener("click", copyJson);
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") analyze();
  if (event.key === "Escape" && isAnalyzing) controller?.abort();
});
new ResizeObserver(() => drawBoxes(latestResult?.objects || [])).observe(elements.image_stage);

evaluationPanel = initializeEvaluationPanel({
  getReadiness: () => latestReadiness,
  refreshReadiness,
});
refreshReadiness().catch(() => {});
