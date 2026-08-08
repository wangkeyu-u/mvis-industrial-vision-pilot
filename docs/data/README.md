# 数据与评测里程碑说明

本目录记录视觉合规审查默认场景的数据契约、划分口径和评测口径。实现只依赖 Python 标准库与 Pillow，不加载模型，适合在 M5 MacBook Air 16GB 上执行全量数据预检。

## 核心契约

- `src.data.DataSample.from_dict` 是单样本的规范化入口。每个样本必须提供 `sample_id`、`entity_id`、相对图片路径、指令、标准响应、难度标签、数据来源和结构化许可证。
- 边界框是原图像素坐标 `[x1, y1, x2, y2]`，右/下边界不超过宽/高，且 `x1 < x2`、`y1 < y2`。
- `license.identifier` 是 manifest 许可证注册表的稳定键。`redistributable`、`commercial_use` 和 `derivative_work` 使用可空布尔值：`null` 表示未审查，不得在发布时解读为允许。
- `DatasetManifest` 集中记录数据版本、SHA-256、dHash、实体、划分、许可证、冻结状态和版本差异，通过 `write_atomic` 原子写入。

JSON Schema 位于 `configs/data/sample.schema.json` 和 `configs/data/manifest.schema.json`；Python 实现还执行 JSON Schema 不能表达的跨字段约束。

## 稳定调用接口

- 数据构建：`src.data.prepare_dataset(samples, dataset_root, dataset_version=..., created_at=...)`。返回 `DatasetPreparationResult`，同时包含校验报告、感知哈希、近重复组、划分表、泄漏报告和 manifest。
- JSONL 加载：`src.data.load_samples_jsonl(path)`，错误会指向具体行号。
- 独立泄漏审计：`src.data.detect_split_leakage(samples, assignments, duplicate_groups)`，可用于审计外部划分；`LeakageReport.passes(0.005)` 执行发布门禁。
- 统一评测：`src.evaluation.evaluate_cases(cases)`。返回 `EvaluationSummary`，五项指标与 `sample_count`、`localization_count`、`hard_negative_count` 同时提供。
- 系统输出适配：`normalize_model_result(result)` 直接接收算法 `ModelResult`；`normalize_analyze_response(payload)` 接收 `/v1/analyze` 解析后的 mapping 或原始 JSON 字符串。
- 批量评测：`evaluate_offline_records(records)` 输出逐样本结果、五项指标、切片指标和失败案例；机器报告契约位于 `configs/eval/report.schema.json`。
- KSDD V0：`prepare_ksdd_dataset(...)` 将官方精细标注发布转换为实体/近重复隔离的冻结 pilot；命令行入口为 `python -m src.data.ksdd_cli`。真实数据证据、许可和限制见 [phase5_dataset_report.md](phase5_dataset_report.md)。

这些门面仅接收标准 Python 对象或本地文件，不依赖模型、API 或网络数据。

## 批量离线评测 CLI

```bash
python -m src.evaluation.cli \
  --ground-truth tests/evaluation/fixtures/system_ground_truth.jsonl \
  --predictions tests/evaluation/fixtures/system_predictions.jsonl \
  --output artifacts/evaluation/report.json \
  --iou-threshold 0.5
```

真值 JSONL 每行必须包含 `sample_id`、`result`、`objects`、`image_width`、`image_height`，可选 `is_hard_negative` 和 `slices`。也可直接使用带 `response`、`difficulty`、`source` 的标准数据样本，但仍需补充原图宽高。报告会另外自动生成 `prediction_source:api`、`prediction_source:model_result` 等输出链路切片。

预测 JSONL 每行格式为：

```json
{"sample_id":"s1","source":"api","prediction":{"schema_version":"1.0.0","request_id":"req-1","model":{"base":"active","adapter":null},"result":"compliant","objects":[],"reason":"no target","uncertain":false,"latency_ms":100,"warnings":[]}}
```

`source` 支持 `api` 和 `model_result`。为保留无效模型输出用于 JSON 有效率评测，可用 `prediction_raw` 字符串代替 `prediction`。缺失预测会记为无效失败，不会被静默过滤；多出的未知 `sample_id` 会阻断评测。

算法代码可直接适配已生成的 `ModelResult`：

```python
from src.evaluation import normalize_model_result

model_result = adapter.analyze(model_request)
prediction = normalize_model_result(model_result)
```

后端批量采集可将每个 `/v1/analyze` 响应包装为一行 JSONL，再运行上述 CLI。例如：

```bash
response_json="$(curl -sS -X POST http://127.0.0.1:8000/v1/analyze \
  -F 'image=@fixture.jpg;type=image/jpeg' \
  -F 'query=找出不符合要求的区域')"
jq -cn --arg sample_id "sample-001" --argjson prediction "$response_json" \
  '{sample_id:$sample_id,source:"api",prediction:$prediction}' >> predictions.jsonl
```

## 可复现评测包与 KPI 验收

在普通报告参数后增加以下参数即可导出评测包：

```bash
python -m src.evaluation.cli \
  --ground-truth tests/evaluation/fixtures/system_ground_truth.jsonl \
  --predictions tests/evaluation/fixtures/system_predictions.jsonl \
  --output /tmp/mvis-fixture-report.json \
  --package-dir /tmp/mvis-fixture-package \
  --data-manifest tests/evaluation/fixtures/dataset_manifest.json \
  --baseline-metrics tests/evaluation/fixtures/baseline_metrics.json \
  --created-at 2026-08-08T00:00:00Z \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 20260808 \
  --fixture-only
```

评测包目录包含：

- `config.json`：解析后的评测参数、KPI 阈值和 bootstrap 参数；
- `data_provenance.json`：数据版本、冻结状态和原 manifest SHA-256；
- `metrics.json`：指标、置信区间、KPI-01～05 判定及未满足原因；
- `failures.json`：按失败类型和切片聚合的错误样本；
- `environment.json`：Python、系统、依赖、Git commit 和工作树状态；
- `report.json`：完整逐样本评测报告；
- `report.md` 与 `report.html`：包含指标、区间、KPI、失败案例、切片差异和候选比较的人类可读报告；
- `comparison.json`：可选的公平配对候选比较；
- `package_manifest.json`：全部组件的 SHA-256。

置信区间使用固定 seed 的 percentile bootstrap。Macro-F1 按样本成对重采样；定位按真值目标命中与否重采样；JSON、困难负例和一致性按各自有效分母重采样。空切片的区间边界为 `null`，不解读为达标。

KPI-01 与 KPI-02 同时检查绝对阈值和相对零样本基线的百分点提升；KPI-04 同时检查假阳性率上限和相对下降比例。缺失必需基线时状态为 `not_evaluable`，不会默认通过。

当数据 manifest 明确记录 `formal_kpi_eligible=false` 时，评测包自动添加 `PILOT DATASET` 水印，并把 KPI-01～05 全部设为 `pilot_only`、`eligible_for_model_acceptance=false`。该门禁不能被表面达标的指标值绕过。

`tests/evaluation/fixtures` 下的数据仅用于验证评测代码。对它们导出时必须传入 `--fixture-only`；此时评测包及全部 KPI 状态强制为 `fixture_only`、`eligible_for_model_acceptance=false`，不得作为模型成绩或 MVP 验收证据。

Mock 输出必须传入 `--mock-only`。Markdown 和 HTML 顶部会显示高对比水印，KPI 状态强制为 `mock_only`。

## 领域数据导入与数据卡

```bash
python -m src.data.import_cli \
  --input source_samples.jsonl \
  --dataset-root /path/to/dataset \
  --output canonical_samples.jsonl \
  --manifest-output manifest.json \
  --summary-output import_summary.json \
  --data-card-output DATA_CARD.md \
  --rejects-output rejected_samples.jsonl \
  --license-policy configs/data/license_policy.yaml \
  --domain compliance \
  --dataset-version compliance-1.0.0 \
  --created-at 2026-08-08T00:00:00Z
```

导入 JSONL 接受标准 `DataSample` 或展开的 `result/objects/reason/uncertain` 字段。`entity_id`、图片相对路径、来源和结构化许可证为必需字段；缺失 `sample_id` 时使用领域名与规范化内容哈希生成稳定 ID。

拒绝清单优先于白名单。不在白名单、显式禁止、格式错误、图片损坏或标注越界的样本写入 rejection JSONL。默认任何拒绝都使数据发布返回码 2；只有明确传入 `--allow-partial` 才会用通过样本继续。

导入摘要会统计结论、来源、许可证、困难负例、难度标签、自定义切片、划分和近重复泄漏。数据卡在样本少于 300 或困难负例低于 20% 时自动增加限制说明；合成数据应传 `--fixture-only` 生成水印。

## 候选模型公平比较

```bash
python -m src.evaluation.compare_cli \
  --ground-truth frozen_test.jsonl \
  --baseline-predictions zero_shot_predictions.jsonl \
  --candidate-predictions candidate_predictions.jsonl \
  --output comparison.json \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 20260808 \
  --minimum-reliable-samples 30
```

比较前强制核对样本 ID、逐样本真值和 IoU 阈值。五项指标使用配对 bootstrap，FPR 以“越低越好”转换为 improvement delta。任一指标的有效分母少于 30 时，整个比较降级为 `degraded_small_sample`，提示固定为 `insufficient_sample`，不得声称统计显著。切片差异未做多重比较校正，仅作描述。

## 质量检查和划分

`validate_image` 同时检查文件大小、文件签名、实际解码格式、像素总数和完整解码。`validate_dataset` 在此基础上检查重复 ID、路径越界、标注框越界以及困难负例含正例框。

去重使用 64-bit dHash，默认汉明距离不超过 5 的图片归入近重复连通分量。`entity_isolated_split` 会先合并相同 `entity_id` 和近重复分量，再按固定种子贪心分配。因分量不可拆分，小数据集的实际比例可能偏离 70/15/15，必须在 manifest 统计中报告实际数量。

## 评测口径

- Macro-F1：真值和预测出现的类别的逐类 F1 算术平均；显式传入的零支持类别记 0。
- Acc@IoU0.5：每个 grounding 查询对齐一个真值框，IoU `>= 0.5` 命中，缺失预测记为未命中。多目标样本在评测适配层拆成每真值目标一条对齐记录。
- JSON 有效率：未经修复的原始输出能解析，且通过结构字段、类型、bbox 和置信度检查的比例。
- 困难负例假阳性率：仅在 `hard_negative` 切片上计算；肯定违规结论、任何证据框或无法解析的输出均记假阳性。
- 证据-结论一致率：违规结论必须有框且不确定标志为假；合规结论必须无框；不确定结论必须 `uncertain=true` 且无框。无效 JSON 记为不一致。

所有空切片指标返回 `0.0`，正式报告必须同时输出分母样本数，避免将无样本误报为达标。

## 当前风险

- dHash 是快速候选召回方法，不能代替人工复核；截剪、局部拼接和大幅文字叠加可能逃逸。
- 数据版本只校验格式，“冻结后修改必须升版”需要后续 CI 将当前 manifest 与已发布 manifest 比对。
- 许可证三个布尔权限为 `null` 时代表未审查，发布门禁必须将其视为阻断项。
