# 算法实验闭环

状态：`dry-run-validated / real-experiments-not_run`

## 实验矩阵

`generate_default_matrix` 生成 24 个预注册实验，覆盖零样本、LoRA、QLoRA、4/8/16-bit 与两档分辨率、Florence-2/RF-DETR 独立基线以及单/双 specialist 融合。矩阵生成不代表实验已执行；每个条目都附带 `status=not_run` 的 `RunManifest`。

校验当前配置且明确显示数据集未就绪：

```bash
python3 -m src.training.experiment_cli \
  --config configs/models/qwen3_vl_2b_mlx_4bit.json \
  --require-runnable
```

当冻结数据版本确定后，只进行 dry-run readiness 校验：

```bash
python3 -m src.training.experiment_cli \
  --config configs/models/qwen3_vl_2b_mlx_4bit.json \
  --data-version domain-v1 \
  --require-runnable > /tmp/experiment-matrix.json
```

dry-run 还会检查固定 revision 的本地权重快照、量化制品、LoRA/QLoRA 执行能力和 specialist 实现。只有已核验的能力才可用 `--available-quantization-bit`、`--available-adaptation` 或 `--available-specialist` 显式声明。`runnable=true` 仍只表示静态前置条件满足，不表示训练成功或指标达标。

## 融合策略

`ConservativeFusionPolicy` 使用高置信 specialist 候选和 IoU 匹配：

- 主 VLM 违规证据与 specialist 全量匹配时，输出 `source=fusion` 的证据；
- 合规与 specialist 正例冲突、框不重叠或证据数不一致时，输出 `MODEL_CONFLICT` 拒答；
- 主 VLM 低置信且无佐证时，输出 `LOW_CONFIDENCE` 拒答；
- 主 VLM 已拒答且 specialist 无高置信证据时，保留原拒答；
- 多 specialist 模式要求所有来源都与主 VLM 一致，任一来源冲突即转人工复核；
- 结果在 `provenance.extra` 保留融合策略、specialist 来源、候选数和匹配数。

## 预测导出

`export_predictions_jsonl(adapter, samples, destination)` 接收 `BatchPredictionSample` 批次，原子写入 evaluator 可直接读取的 `source=model_result` JSONL。单样本失败记为 `status=unavailable` 且故意不伪装成合法预测；evaluator 会将其计为 schema-invalid。`fail_fast=True` 时不会覆盖旧导出文件。

```python
from src.inference import BatchPredictionSample, export_predictions_jsonl

summary = export_predictions_jsonl(adapter, samples, "predictions.jsonl")
```

## 作品集证据

- [模型卡模板](templates/model_card_template.md)
- [实验记录 JSON 模板](templates/experiment_record_template.json)

实验完成后才能把对应 `EvidenceValue` 从 `not_run` 改为 `recorded`，且必须同时提供数值与可定位的 artifact。无法获取的证据改为 `unavailable` 并填写 reason。
