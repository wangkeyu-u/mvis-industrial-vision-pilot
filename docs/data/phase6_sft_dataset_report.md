# 第六阶段：KSDD SFT V1 冻结报告

## 结论

已从冻结的 `ksdd-0.1.0` 实体隔离划分确定性导出 `ksdd_sft-1.0.0`，供
MLX-VLM/Qwen2/3-VL 单图 SFT 使用。导出包含 `train.jsonl`、`valid.jsonl` 和 `test.jsonl`，
每个源样本只出现一次，没有扩充、复制或从 test 派生训练样本。

本阶段只构建并验证数据、prompt、答案、文件哈希、泄漏门禁和公平评测编排；没有加载模型、
没有运行训练或推理，所有模型指标为 `null`。冻结 test 只有 56 张，因此数据和后续真实评测包
均为 `pilot_only`，不得作为正式 KPI 验收。

## 冻结身份

| 字段 | 值 |
|---|---|
| SFT dataset version | `ksdd_sft-1.0.0` |
| Prompt version | `ksdd_audit_prompt-1.0.0` |
| Source dataset | `ksdd-0.1.0` |
| Source manifest SHA-256 | `fe9983728d7aa830f11a1369ff1a4eb353be8e8403685a1cf45ea23db128085a` |
| SFT manifest SHA-256 | `61c69a560c3d5be0c98ebc68f6e45908094e6a1bde7b6d7d07d0a66d08bdbca1` |
| Created at | `2026-08-08T23:43:09Z` |
| Pilot only | `true` |
| Model/training run | `false` |

追踪锁为 `configs/data/ksdd_sft_v1.lock.json`。完整文件哈希表在
`data/processed/ksdd_sft_v1/sft_manifest.json`，包含 383 张图、三个 JSONL、两份 Schema、
评测真值、prompt catalog、泄漏报告和 readiness report。

## MLX-VLM 官方格式兼容

[MLX-VLM 官方 LoRA 文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/LORA.MD)
要求视觉 SFT 数据包含 `images` 与 `messages`；Qwen2/3-VL 的 user content 使用 image/text
列表，assistant content 使用 text。导出记录严格采用该结构：

```json
{
  "images": ["data/processed/ksdd_sft_v1/images/kos10/Part3.jpg"],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data/processed/ksdd_sft_v1/images/kos10/Part3.jpg"},
        {"type": "text", "text": "<frozen bilingual audit prompt>"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "{\"result\":\"violation\",...}"}
      ]
    }
  ]
}
```

[Hugging Face Datasets 官方加载文档](https://huggingface.co/docs/datasets/en/package_reference/loading_methods)
支持把只含 JSON/JSONL 数据文件的本地目录直接传给 `load_dataset(path, split=...)`。本项目
mlx-vlm 0.6.10 的已安装源码同样调用 `load_dataset(args.dataset, split=args.split)`，所以训练
数据目录参数为 `data/processed/ksdd_sft_v1/hf`。

记录 Schema：

- `configs/data/ksdd_sft_record.schema.json`：锁定 MLX-VLM `images/messages` 结构、版本、split 和 prompt 字段；
- `configs/data/ksdd_sft_answer.schema.json`：锁定严格 JSON 答案；compliant 必须 `objects=[]`，violation 必须包含 bbox，`uncertain=false`。

当前 `.venv` 安装了 mlx-vlm 0.6.10，但没有安装其可选 `datasets` 训练依赖；因此本阶段只验证
记录契约和导出闭环，不执行 `load_dataset`、模型加载或训练。正式训练前需在算法负责人控制的环境中
安装官方 `mlx-vlm[train]` 依赖并登记环境哈希，不应由数据导出过程静默改变项目依赖。

## Prompt 与答案策略

所有 prompt 同时包含英文和中文，并明确：

- 只返回一个 JSON 对象，不得输出 Markdown/code fence；
- 正例 `result=violation`，对象标签为 `surface_defect`；
- bbox 使用原图整数 `[x1,y1,x2,y2]`；
- 负例 `result=compliant` 且 `objects=[]`；
- `uncertain=false`。

`prompt_version` 固定为 `ksdd_audit_prompt-1.0.0`。训练集允许三种语义等价、受控改写；每个
训练样本通过 `SHA-256(sample_id)` 确定唯一变体，不随机、不增加样本：

| Split | Prompt variant | 数量 |
|---|---|---:|
| train | `bilingual_audit_direct` | 84 |
| train | `bilingual_audit_quality_gate` | 88 |
| train | `bilingual_audit_evidence_first` | 99 |
| valid | `bilingual_audit_direct` | 56 |
| test | `bilingual_audit_direct` | 56 |

valid/test 不做改写。test 的 prompt variant、完整文本和 SHA-256
`792f645df4044cc0ed6a236f81db884f714de4368d5f976802de7ff2da74901e` 均被冻结。

答案不使用模型、OCR、规则猜测或合成框：

- 正例只读取 V0 中真实 mask 派生的冻结 bbox；
- 负例只输出 `compliant`、`objects=[]`；
- 双语 reason 为确定性模板，不改变标签；
- 每条 assistant text 都重新 `json.loads`，并与源 `DataSample` 逐字段比对；
- bbox 重新检查原图尺寸边界。

`strict_answer_json_validity=1.0` 是 383 条训练标签文件的结构校验率，不是模型 JSON 有效率，
不得当作模型成绩。

## 样本与文件统计

| Split | 实体隔离来源 | 总数 | violation | compliant | 文件 SHA-256 |
|---|---|---:|---:|---:|---|
| train | source train | 271 | 34 | 237 | `5ca5af6c2aec284f909fc55c017414df5ab026e5fa85e3b1a1c3a34884000097` |
| valid | source validation | 56 | 7 | 49 | `3c2c6d487900c5394d571f78debf48c630ecb9a073374ed1eba60b4bc258f888` |
| test | frozen source test | 56 | 9 | 47 | `6c536189927fd9e487d61252f3922a0be625614374572094e6308dba93a5aada` |

评测真值 `evaluation_ground_truth.jsonl` SHA-256：
`bd6094421831b8b508e09b7f85d76d59b35781b06cc208e0927b274b6c2adde5`。

全部图片从 V0 复制到 SFT 包，逐文件 SHA-256 与源 manifest 比对后才写入。输出目录约 94 MiB，
位于仓库已 ignore 的 `data/processed/ksdd_sft_v1/`。

## 泄漏门禁

导出前重新从 V0 manifest 的实体、dHash 和 split 构造泄漏报告。实际结果：

- source assignments preserved：`true`；
- train/test sample ID overlap：0；
- train/test entity overlap：0；
- train/test exact image SHA-256 overlap：0；
- 近重复跨 split 泄漏率：0.0；
- test-derived training samples：0；
- test prompt fixed：`true`。

导出器不会重新划分，也没有“回填困难负例”“prompt augmentation 后复制到 train”等旁路。
任一源 manifest 哈希变化、图像哈希变化、实体/近重复泄漏、答案与真值不一致或 test prompt 变化都会阻断导出。

## 导出命令

```bash
.venv/bin/python -m src.data.ksdd_sft_cli \
  --source-root data/processed/ksdd_v0 \
  --source-lock configs/data/ksdd_v0.lock.json \
  --output-root data/processed/ksdd_sft_v1 \
  --record-schema configs/data/ksdd_sft_record.schema.json \
  --answer-schema configs/data/ksdd_sft_answer.schema.json \
  --created-at 2026-08-08T23:43:09Z \
  --reference-root .
```

稳定 Python 接口：

```python
from src.data import export_ksdd_sft, load_sft_jsonl, validate_sft_record
```

## LoRA 训练命令（只准备，未执行）

以下命令与 mlx-vlm 0.6.10 官方参数相符；M5 16GB 首轮建议 batch size 1、梯度检查点和只训练
assistant completion。它没有在本阶段执行：

```bash
MODEL_SNAPSHOT="$(.venv/bin/python -c \
  'from huggingface_hub import snapshot_download; print(snapshot_download("mlx-community/Qwen3-VL-2B-Instruct-4bit", revision="9c4f5209e57b31f4b9dfba735de3fb983739c9cc", local_files_only=True))')"
.venv/bin/python -m mlx_vlm.lora \
  --model-path "$MODEL_SNAPSHOT" \
  --dataset data/processed/ksdd_sft_v1/hf \
  --split train \
  --train-mode sft \
  --train-on-completions \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --epochs 2 \
  --learning-rate 2e-5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --grad-checkpoint \
  --max-seq-length 1024 \
  --output-path ARTIFACTS/ksdd_sft_v1_lora
```

注意：已安装 mlx-vlm 0.6.10 的 CLI 当前把 `val_dataset=None` 传给 trainer。`valid.jsonl` 已冻结，
但不能假定上述 CLI 会自动报告 valid loss 或据此选 checkpoint；正式训练需先确认训练器版本的验证集接线，
不得用 test loss 代替 validation。

## 零样本与 LoRA 公平评测（只准备，未执行）

两次预测必须使用同一：

- 56 个冻结 test sample ID；
- `ksdd_audit_prompt-1.0.0` 与固定 direct prompt；
- 基础模型 revision `9c4f5209e57b31f4b9dfba735de3fb983739c9cc`；
- 模型配置 SHA-256 `ef76aa6b9a49d5a1ea1066907c31e85cb914bb2c5726096042c89a019ed45cf1`；
- seed `20260809`、greedy、temperature 0、top-p 1、max tokens 256；
- 推理适配、JSON 解析、bbox 坐标处理和失败计数规则。

唯一允许差异是 LoRA adapter 是否启用。预测文件必须完整覆盖 56 个 ID，缺失或额外 ID 都会阻断比较。
两份运行身份文件遵循 `configs/eval/ksdd_sft_run.schema.json`：zero-shot 的 adapter 必须为
`{"enabled":false,"sha256":null}`，LoRA 必须登记真实 adapter SHA-256。

得到 `ZERO_SHOT.jsonl` 和 `LORA.jsonl` 后运行：

```bash
.venv/bin/python -m src.evaluation.sft_fair_cli \
  --ground-truth data/processed/ksdd_sft_v1/evaluation_ground_truth.jsonl \
  --zero-shot-predictions ZERO_SHOT.jsonl \
  --lora-predictions LORA.jsonl \
  --zero-shot-run-config ZERO_SHOT_RUN.json \
  --lora-run-config LORA_RUN.json \
  --data-manifest data/processed/ksdd_sft_v1/sft_manifest.json \
  --output-dir EVAL_OUTPUT \
  --created-at ISO8601 \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 20260809 \
  --minimum-reliable-samples 30
```

该命令会严格比较运行身份与预测覆盖，输出两份逐样本报告、两份可复现评测包和 paired bootstrap
比较。SFT manifest 会让 KPI-01～05 强制为 `pilot_only`。test 中只有 9 个定位正例，小于 30，
因此定位相关比较必然降级为 `degraded_small_sample/insufficient_sample`，不得声称统计显著。

当前机器可读状态在 `pilot_readiness_report.json`：`model_run=false`，zero-shot、LoRA 和 comparison
指标全部为 `null`。这份 readiness report 是命令与门禁准备证据，不是模型评测报告。

## 验证证据

新增测试覆盖：

- V0 manifest 错误哈希阻断；
- 官方 `images/messages` 结构；
- prompt 变体确定性与 test 单版本；
- 正负例严格 JSON、bbox 边界与源真值对齐；
- train/test ID、实体、图像 SHA 和近重复零泄漏；
- 文件哈希与 manifest 复算；
- zero-shot/LoRA 运行身份只允许 adapter 差异；
- 预测必须完整覆盖冻结 test；
- fixture 公平比较保持 fixture 水印，不冒充模型成绩。

最终回归命令：

```bash
.venv/bin/ruff check src/data src/evaluation tests/data tests/evaluation
.venv/bin/python -m pytest tests/data tests/evaluation -q
```

实际结果：静态检查 `All checks passed!`，数据/评测回归 `54 passed in 0.69s`。独立完整性复核
重算了 392 个登记文件哈希，并逐条验证 383 个 SFT record、图片路径、双语 prompt、严格 JSON
答案、原图 bbox、源 split 映射和 56 条 test 评测真值，全部通过。

## 风险

- KSDD 与 SFT 改编继续受 CC BY-NC-SA 4.0 非商业、署名、相同方式共享限制；商业训练需另行授权。
- 数据仅覆盖受控灰度换向器表面，不能代表开放场景或通用视觉问答。
- 50 个正例、test 9 个定位正例太少，任何指标区间都会很宽。
- 三种训练 prompt 都是人工受控模板，可能产生模板过拟合；固定 test prompt 只能保证同版本公平，不能测 prompt 鲁棒性。
- bbox 是 mask 的联合外接框，丢失像素形状与多连通区域细节。
- MLX-VLM 训练依赖尚未安装，LoRA/QLoRA 在 M5 16GB 上的显存、收敛、速度和 checkpoint 可用性均未验证。
- JSONL 中图像路径相对项目根目录；训练命令必须从项目根执行，或在迁移后重新导出并升版。
