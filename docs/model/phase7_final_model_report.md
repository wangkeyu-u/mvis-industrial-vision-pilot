# 第七阶段算法救援与最终候选报告

## 结论

第七阶段已完成两条真实路线、冻结 56 图公平评测、运行时桥接和全量回归。最终选择 **Anomalib PatchCore + ResNet-18** 作为 `pilot_candidate`，不是生产就绪模型。

选择依据是同一 KSDD V0 冻结 test 上真实宏 F1 最高且资源合格：PatchCore 为 `0.78125`，tile QLoRA 为 `0.45631`，零样本 VLM 为 `0.0`。PatchCore 的 P50/P95 为 `7.312/15.145 ms`，MPS driver 峰值 `1066.703 MB`。但它在 9 个定位目标上 `Acc@IoU=0`，只能作为分类/热力图 pilot，不能宣称精确定位通过。

本轮没有重复 OOM、没有下载第二个 backbone、没有用测试标签校准阈值，也没有把未运行结果写成成功。

## 冻结输入与可复现身份

- KSDD 数据版本：`ksdd-0.1.0`
- 冻结 manifest SHA-256：`fe9983728d7aa830f11a1369ff1a4eb353be8e8403685a1cf45ea23db128085a`
- test：56 张，其中 violation 9、compliant/困难负例 47
- 标准 SFT manifest SHA-256：`61c69a560c3d5be0c98ebc68f6e45908094e6a1bde7b6d7d07d0a66d08bdbca1`
- tile 256 manifest SHA-256：`2cf3d19b5b22baed19a2d75572b4243dc151258e32d20a23ac0c73222803e11f`
- tile 384 manifest SHA-256：`542873a593e6b397742ebc4928bc13222796bb95c4aba8c5f8b4e72fecd19247`
- tile split：train/valid/test = `813/168/168`，每张原图固定 top/middle/bottom 三个 500×500 tile；test-derived training count = `0`

tile 视图是从数据线程发布的标准 SFT 包确定性派生的运行产物，保留原 train/valid/test source sample 分配。它被标记为 `formal_kpi_eligible=false`，最终 KPI 始终在原 56 张 test 上计算。

## 路线 A：受控 tile QLoRA

### 固定配置

| 项目 | 值 |
|---|---:|
| 基础模型 | `mlx-community/Qwen3-VL-2B-Instruct-4bit` |
| revision | `9c4f5209e57b31f4b9dfba735de3fb983739c9cc` |
| 量化 | 4-bit，group size 64，affine |
| seed | `20260808` |
| rank / alpha / dropout | `4 / 8 / 0.05` |
| batch / gradient accumulation | `1 / 4` |
| 学习率 | `1e-4` |
| 视觉编码器 | frozen；只训练 language LoRA，4.358144M trainable params |
| 最大输出 | 128 tokens |
| 最大序列 | 512 |
| 训练内存门禁 | 12288 MB |

MLX-VLM 官方训练 API 实际调用 `setup_model_for_training`、`VisionDataset` 和 `sft_trainer.train`；没有另写一个不能代表正式运行的训练器。

### 分级门禁结果

| 档位 | 运行 | Train loss | Val loss | MLX 峰值 | wall time | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 256 | 1 step probe | 3.2357 | 3.301 | 2493.078 MB | 7.178 s | 通过 |
| 384 | 1 step probe | 3.2089 | 3.269 | 3666.918 MB | 6.649 s | 通过 |
| 256 | 20 step smoke | 1.4336 | 1.043 | 2841.929 MB | 25.881 s | 通过 |
| 384 | 20 step smoke | 1.4298 | 1.037 | 3741.039 MB | 32.374 s | 通过并入选正式训练 |
| 384 | 100 step candidate | 0.00147 | 0.001 | 3741.039 MB | 155.354 s | 完成 |

384 档仅因 smoke 验证损失略优而进入唯一一次正式候选训练。正式训练耗时 2.59 分钟，远低于 60 分钟门禁；没有 OOM 或重试。

候选 adapter：

- SHA-256：`69b4ff0ea1ee7b313ef1276d762c92c13cd02b8e6c01095237b8bf7848eb9275`
- training log SHA-256：`43dc882a639ad5854e06a3f88f45269ebe0a090bf5c233b26891b8d5ff5e2b12`
- 权重位于 `artifacts/model/phase7/qlora_candidate_384/adapter/`，不提交版本库。

### 冻结 56 图真实评测

对每张原图运行 3 个 384 tile，共 168 次真实 MLX 推理，再将 tile bbox 反变换到原图坐标并聚合成严格 56 行 evaluator JSONL。

| 指标 | 结果 |
|---|---:|
| 完成/不可用 | 56 / 0 |
| JSON 结构有效率 | 1.0 |
| evidence/conclusion 一致率 | 1.0 |
| 宏 F1 | 0.4563106796 |
| 困难负例 FPR | 0.0 |
| Acc@IoU 0.5 | 0.0 |
| P50 / P95（每原图三 tile 总和） | 3029.859 / 4127.119 ms |
| MLX 峰值 | 2808.791 MB |
| process peak RSS | 2406.938 MB |

失败模式明确：候选对 56 张全部输出 compliant，47 个负例正确，但 9 个正例全部漏检，因此训练/验证 loss 的明显下降没有转化为正类召回或定位能力。该 adapter 只证明 tile QLoRA 在机器上可受控运行，不作为最终候选。

## 路线 B：工业异常 specialist

### 选择与下载约束

优先项 EfficientAD 没有执行，因为其官方流程还会自动取得 teacher 权重和 ImageNette 数据，与“只下载一个受控 backbone”的本轮约束冲突。回退到官方 Anomalib PatchCore，并只使用一个较小的 `timm/resnet18.a1_in1k` backbone。

- Anomalib：2.3.3，Apache-2.0
- torch / torchvision / timm：2.13.0 / 0.28.0 / 1.0.28
- backbone revision：`491b427b45c94c7fb0e78b5474cc919aff584bbf`
- backbone 大小：46,807,446 bytes
- backbone SHA-256：`80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`
- 额外模型下载：0；总受控 backbone 数：1；远低于 3 GB 门禁

### 拟合与校准

- 只使用冻结 train 中 237 张 compliant 图拟合 memory bank。
- layer2/layer3，image size 256，coreset ratio 0.001，memory bank `242×384`。
- 特征提取 1.780 s，coreset 3.025 s，总拟合 4.805 s。
- 分类阈值 `13.096867561340332`、bbox threshold `0.8` 均只用 validation 确定；`test_labels_used=false`。
- validation 宏 F1 `0.84878`，accuracy `0.94643`；validation 定位 Acc@IoU `0.14286`、mean IoU `0.18491`。

checkpoint SHA-256：`1ad6587c26805639d9dc6c323a468f0da113f48e2d8e481ae5d0ab595734a681`。

### 冻结 56 图真实评测

| 指标 | 结果 |
|---|---:|
| JSON 结构有效率 | 1.0 |
| evidence/conclusion 一致率 | 1.0 |
| 宏 F1 | 0.78125 |
| 分类正确 | 50 / 56 |
| 困难负例 FPR | 0.042553（2 / 47） |
| Acc@IoU 0.5 | 0.0（0 / 9） |
| P50 / P95 | 7.312 / 15.145 ms |
| MPS driver peak observed | 1066.703 MB |
| process peak RSS | 1914.797 MB |
| 完整运行 wall time | 7.898 s |

主要失败：2 个困难负例假阳性、4 个正例分类漏检；分类为正的定位框也没有达到 IoU 0.5。热力图和原图 bbox 已全部真实输出，但 bbox 只能用于 pilot 可视化/复核，不能作为自动精确定位证据。

## 三路线比较与最终选择

| 路线 | 宏 F1 | JSON 有效率 | 困难负例 FPR | Acc@IoU | P50 / P95 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 零样本 Qwen3-VL | 0.0 | 0.98214 | 0.0 | 0.0 | 1882 / 2308 ms | 拒答过多，不可用 |
| tile QLoRA 384 | 0.45631 | 1.0 | 0.0 | 0.0 | 3030 / 4127 ms | 全负预测，正类不可用 |
| PatchCore ResNet-18 | **0.78125** | 1.0 | 0.04255 | 0.0 | **7.312 / 15.145 ms** | 最终 pilot 候选 |

选择使用了这 56 张 test 的标签，所以该比较是候选选择证据，不是独立、无偏的最终验收。进入 production 之前必须在新的冻结 holdout 上复验，尤其要重新设计定位提取；当前只允许 `pilot_candidate`。

## 后端 specialist 运行时桥

工厂：

```python
from src.inference.service_bridge import create_specialist_service_adapter

adapter = create_specialist_service_adapter(
    "artifacts/model/phase7/patchcore_resnet18/run_manifest.json"
)
assert adapter.ready
output = await adapter.detect(specialist_request)
```

工厂加载时执行：completed/schema/algorithm 校验、checkpoint 双 SHA 校验、checkpoint config 与 dataset identity 校验、验证集阈值校验、单 backbone 缓存 SHA 校验，并在 import Anomalib 前强制 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。缺少或篡改任何工件都会失败，绝不下载。

输入 `SpecialistRequest` 字段：`image_bytes`、`image(width,height,format,mode,byte_size)`、`request_id`、`query`、`task`、`options`。

输出 `SpecialistOutput` 字段：

- `score: float`：真实图像异常分数；
- `threshold: float`：validation 固定阈值；
- `source: "patchcore"`；
- `objects[]`：阳性时包含 `surface_defect`、原图坐标 bbox、confidence、source；
- `heatmap_png: bytes`：按原图宽高输出的灰度 PNG，由 API 的有界 artifact store 接管；
- `warnings[]`：包含 `patchcore_validation_calibrated`。

`detect` 使用 `asyncio.to_thread`，不捕获 `CancelledError`，取消会原样传播；Torch 推理由锁串行，避免 MPS 共享状态并发冲突。

## 许可证与使用限制

| 资产 | 记录 |
|---|---|
| Anomalib | Apache-2.0，官方仓库 `open-edge-platform/anomalib` |
| timm 代码与 `resnet18.a1_in1k` 模型卡 | Apache-2.0 |
| ImageNet 预训练权重 | timm 明确提示原 ImageNet 数据集的非商业研究条款可能延续到权重；商业使用需法律复核 |
| Qwen3-VL 基础与 MLX 转换 | Apache-2.0；模型身份沿用 phase 5 已核验记录 |
| KSDD | CC-BY-NC-SA-4.0；因此当前实验与产物不是商业生产许可证明 |

许可证结论：技术指标即使后续通过，也不能自动解除 KSDD 和 ImageNet 权重的商业使用风险。

## 工件

- 比较：`artifacts/model/phase7/comparison.json`，SHA-256 `23da8063498f3a0ee5e8ee2c67a84df42666dd398abef67df13d35818ae3b2b2`
- QLoRA 预测：`artifacts/model/phase7/qlora_candidate_384_test/predictions.jsonl`
- QLoRA 评估：`artifacts/model/phase7/qlora_candidate_384_test/evaluation_report.json`
- PatchCore manifest：`artifacts/model/phase7/patchcore_resnet18/run_manifest.json`
- PatchCore checkpoint：`artifacts/model/phase7/patchcore_resnet18/patchcore_resnet18.pt`
- PatchCore 预测/分数/热力图：同目录 `predictions.jsonl`、`raw_scores.jsonl`、`heatmaps/`

## 执行命令与测试证据

```bash
.venv/bin/python -m src.training.tile_sft --source-root data/processed/ksdd_sft_v1 --output-root artifacts/model/phase7/ksdd_tile_sft_384 --image-size 384
.venv/bin/python -m src.training.mlx_vlm_runner --training-config configs/models/ksdd_tile_qlora_384_phase7.json --model-config configs/models/qwen3_vl_2b_mlx_4bit.json --sft-root artifacts/model/phase7/ksdd_tile_sft_384 --output-dir artifacts/model/phase7/qlora_probe_384 --steps 1 --run-kind probe --validation-batches 1
.venv/bin/python -m src.training.mlx_vlm_runner --training-config configs/models/ksdd_tile_qlora_384_phase7.json --model-config configs/models/qwen3_vl_2b_mlx_4bit.json --sft-root artifacts/model/phase7/ksdd_tile_sft_384 --output-dir artifacts/model/phase7/qlora_smoke_384 --steps 20 --run-kind smoke --validation-batches 2
.venv/bin/python -m src.training.mlx_vlm_runner --training-config configs/models/ksdd_tile_qlora_384_phase7.json --model-config configs/models/qwen3_vl_2b_mlx_4bit.json --sft-root artifacts/model/phase7/ksdd_tile_sft_384 --output-dir artifacts/model/phase7/qlora_candidate_384 --steps 100 --run-kind formal --validation-batches 4
.venv/bin/python -m src.inference.tile_lora_eval --model-config configs/models/qwen3_vl_2b_mlx_4bit.json --adapter-path artifacts/model/phase7/qlora_candidate_384/adapter --tile-sft-root artifacts/model/phase7/ksdd_tile_sft_384 --output-dir artifacts/model/phase7/qlora_candidate_384_test
.venv/bin/python -m src.inference.patchcore_specialist --config configs/models/ksdd_patchcore_resnet18_phase7.json --dataset-root data/processed/ksdd_v0 --output-dir artifacts/model/phase7/patchcore_resnet18
.venv/bin/python -m pytest -q tests/model
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/inference src/training configs/models tests/model docs/model
```

最终验证：

- `tests/model`：73 passed，4 subtests passed；43 个 warning 均来自 Torch JIT deprecation。
- 全仓：246 passed，12 subtests passed；44 warnings，无失败。
- ruff：All checks passed。

## 下一阶段实验计划

1. 冻结一个未参与本轮选择的新 holdout，复验 PatchCore 分类指标，避免 test-selection bias。
2. 将当前最大连通域 bbox 改为 validation-only 的多连通域、纵向先验或像素阈值/形态学组合实验，以 `Acc@IoU` 为硬门禁；未提高前不接受自动定位。
3. 对 PatchCore 阈值做 bootstrap 区间与按 physical item 分组验证，确认 2/47 假阳性是否集中在特定纹理/批次。
4. 若继续 VLM 路线，先处理 tile 正负极度不平衡与正类 sampling，再在 validation 上看正类 recall；禁止仅依据 loss 继续扩大步数。
5. 在许可审查完成前，保持 candidate 为 non-commercial pilot，不进入 production serving tier。
