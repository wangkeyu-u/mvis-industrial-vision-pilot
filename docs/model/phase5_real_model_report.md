# Phase 5 Real Model Report

状态：`runtime_ready / real_probe_completed_with_failures / quality_not_accepted`  
执行日期：2026-08-08  
设备：Apple M5 MacBook Air，16GB 统一内存  
场景：视觉合规审查

## 结论

固定 revision 的 Qwen3-VL-2B-Instruct 4-bit MLX 模型已在本机完成下载、哈希登记、离线加载和真实推理。`diagnostics --require-ready` 返回退出码 0，证明依赖、固定快照和 MLX 运行前置条件就绪；mlx-vlm 0.6.10 可直接加载该模型，无需兼容补丁。

这不构成模型质量验收。单图虽然完成结构化解析，但把规则指定的红色矩形误判为 `compliant`。首轮 5 图探针只有 4/5 通过结构化校验，1 个输出重复并在 token 上限处截断。澄清规则后的第二轮 5 图全部被严格适配器拒绝，原因包括越界 bbox、缺失 confidence、非空错误 refusal code 和负例幻觉。当前证据只支持“真实运行链路落地”，不支持“合规模型可用”。

## 下载前门禁与来源

| 项目 | 核验结果 |
|---|---|
| 剩余磁盘 | 266GiB |
| 单模型下载上限 | 6GB |
| API 文件总量 | 1,798,023,774 bytes（约 1.67GiB） |
| 实际缓存占用 | 1.7GiB |
| 官方基础模型 | [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| MLX 转换模型 | [`mlx-community/Qwen3-VL-2B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit) |
| 转换来源声明 | 从官方 Qwen 模型转换，转换工具 mlx-vlm 0.3.4 |
| 许可证 | 官方模型卡与转换模型卡均标记 Apache-2.0 |
| 仓库访问 | public、non-gated |
| 固定 revision | `9c4f5209e57b31f4b9dfba735de3fb983739c9cc` |
| API resolved SHA | 与固定 revision 完全一致 |
| 下载模型数 | 1；未下载其他候选 |

Hugging Face API 在下载前返回 16 个文件；`model.safetensors` 为 1,782,040,921 bytes。转换模型卡给出的 MLX 用法与本项目的 mlx-vlm 加载路径一致。

## 本地运行登记

| 字段 | 实际值 |
|---|---|
| Python | 3.13.13 |
| macOS / 架构 | Darwin 25.5.0 / arm64 |
| MLX | 0.32.0 |
| mlx-vlm | 0.6.10 |
| huggingface-hub | 1.27.0 |
| Pillow | 12.3.0 |
| 模型配置指纹 | `71b87d18c01aeb4d` |
| 权重 SHA-256 | `4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb` |
| 缓存状态 | complete；`config.json` 与 `model.safetensors` 均存在 |
| 下载策略 | 配置仍为 `allow_download=false`；运行时只读取本地固定快照 |
| 独立冷加载 | 7041ms |
| diagnostics | `ready=true`，退出码 0 |

权重位于 Hugging Face 用户缓存，没有写入仓库，也不会提交。

## 输入与固定参数

所有输入均由 Pillow 在临时目录中生成，为 256×256 白底纯色几何图形，无外部图片或第三方素材。单图为红色矩形。5 图依次为：大红矩形、绿色矩形、小红矩形、蓝色矩形、被灰色条遮挡的红色矩形。

固定参数：seed `20260808`、`do_sample=false`、temperature `0.0`、top-p `1.0`。单图和 Attempt A 使用 max tokens `128`；为了判断截断是否为唯一问题，Attempt B 使用 `192`，并保留 Attempt A 原始失败。

初始规则：

> 检查图中是否存在红色矩形危险标记；如存在判定 violation 并给出原图像素 bbox；不存在判定 compliant；证据不足判定 uncertain。

澄清规则：

> 任何红色像素组成的矩形区域本身即为违规，不要求文字、符号或其他危险语义。只要看到红色矩形，必须输出 violation 和原图像素 bbox。

## 单图真实探针

| 字段 | 实际值 |
|---|---|
| 输入 SHA-256 | `933be32dc90e58829e203cb0c9438e44398ee5b43dca6957d9bc65f32fb4e140` |
| 预期 | `violation` |
| 结构化状态 | completed |
| 实际判断 | `compliant` |
| 推理耗时 | 3026.817ms |
| 进程峰值 RSS | 1616.469MB |
| 探针进程加载耗时 | 1411.145ms |

原始输出：

````text
```json
{
  "result": "compliant",
  "objects": [],
  "reason": "图像中存在一个红色矩形，但该矩形为纯色填充，且无任何额外标记或符号，未符合危险标记的典型特征。然而，根据任务要求，若图中存在红色矩形危险标记，则判定为violation。但图像中并未显示任何危险标记，仅有一个红色矩形。因此，判定为compliant。",
  "refusal_code": null,
  "insufficient_evidence": null
}
```
````

适配器允许额外 JSON 字段，因此 `insufficient_evidence` 没有导致失败；判定本身与规则不一致。

## 5 图 Attempt A：初始规则

| 指标 | 实际值 |
|---|---|
| 状态 | `completed_with_failures` |
| 成功 / 失败 | 4 / 1 |
| P50 | 3897.246ms |
| P95 | 4134.351ms |
| min / max | 2232.440 / 4134.351ms |
| 进程峰值 RSS | 1808.516MB |
| 探针进程加载耗时 | 1293.716ms |

| # | 图像 SHA-256 | 预期 | 实际 | 耗时 | 证据状态 |
|---|---|---|---|---:|---|
| 1 | `d8eb0b72…585c5` | violation | compliant | 3897.246ms | 结构化成功、结论错误 |
| 2 | `79f22492…a06de` | compliant | compliant | 2482.562ms | 成功 |
| 3 | `2f949378…dcd7e` | violation | unavailable | 4009.090ms | JSON 截断 |
| 4 | `65fdef0f…002b5` | compliant | compliant | 2232.440ms | 成功 |
| 5 | `89e86929…da6c4` | violation | compliant | 4134.351ms | 结构化成功、结论错误 |

逐样本原始输出：

1. 大红矩形

```json
{
  "result": "compliant",
  "objects": [],
  "reason": "图像中存在一个红色矩形，但该矩形为纯色填充，且无任何标记或符号，无法判断为危险标记。根据标准，若存在红色矩形危险标记，则判定为 violation。然而，该矩形并非危险标记，而是正常图形。因此，判定为 compliant。",
  "refusal_code": null,
  "insufficient_evidence": null
}
```

2. 绿色矩形

```json
{
  "result": "compliant",
  "objects": [],
  "reason": "图像中没有红色矩形危险标记。符合要求。",
  "refusal_code": null,
  "insufficient_evidence": null
}
```

3. 小红矩形，失败原因 `ModelOutputError: model output is not one valid JSON object`。以下是完整的截断文本：

````text
```json
{
  "result": "compliant",
  "objects": [],
  "reason": "图像中存在一个红色的矩形，但该矩形为一个简单的图形，且其位置和形状符合危险标记的特征。然而，根据图像内容，该图形是一个简单的红色矩形，且其位置和形状符合危险标记的特征。因此，该图形是一个简单的红色矩形，且其位置和形状符合危险标记的特征。因此，该图形是一个简单的红色矩形，且其位置和形状符合危险标记的特征。因此，该图形是一个简单的红色矩形，
````

4. 蓝色矩形

```json
{
  "result": "compliant",
  "objects": [],
  "reason": "图像中没有红色矩形危险标记，符合要求。",
  "refusal_code": null,
  "insufficient_evidence": null
}
```

5. 遮挡红色矩形

```json
{
  "result": "compliant",
  "objects": [],
  "reason": "图像中没有红色矩形危险标记。图中有一个灰色的竖直矩形，其余为红色的矩形，但并非危险标记。根据任务要求，检查是否存在红色矩形危险标记，图像中存在红色矩形，但其为正常背景色，非危险标记。因此判定为compliant。",
  "refusal_code": null,
  "insufficient_evidence": null
}
```

## 5 图 Attempt B：规则澄清

Attempt B 不是对 Attempt A 的覆盖，而是提示敏感性诊断。5 个调用均完成生成，但 0/5 通过适配器契约，因此退出码为 2。

| 指标 | 实际值 |
|---|---|
| 状态 | `completed_with_failures` |
| 成功 / 失败 | 0 / 5 |
| P50 | 1308.970ms |
| P95 | 4662.195ms |
| min / max | 1182.458 / 4662.195ms |
| 进程峰值 RSS | 969.109MB |
| 探针进程加载耗时 | 1753.044ms |

这些延迟是“生成尝试”耗时，不是成功端到端延迟；不能用于服务验收。峰值为进程 `ru_maxrss`，不是系统级统一内存峰值，跨独立进程的数值不可直接比较。

| # | 原始判断 | 原始 bbox | 拒绝原因 |
|---|---|---|---|
| 1 | violation | `[54,114,256,819]` | bbox 越过 256×256 原图；非空错误 refusal code |
| 2 | violation（绿色负例幻觉） | `[111,174,256,319]` | bbox 越界；负例结论错误；非空错误 refusal code |
| 3 | violation | `[299,400,414,600]` | bbox 越界且缺失 confidence |
| 4 | violation（蓝色负例幻觉） | `[84,284,240,444]` | bbox 越界；理由与结论自相矛盾；非空错误 refusal code |
| 5 | violation | `[77,155,256,824]` | bbox 越界 |

逐样本原始生成文本：

1. 大红矩形

```text
{"result":"violation","objects":[{"label":"red rectangle","bbox":[54,114,256,819],"confidence":0.99}],"reason":"The image contains a red rectangular region, which is the primary violation as per the inspection criteria.","refusal_code":"violation","refusal_code":"violation"}
```

2. 绿色矩形

```text
{"result":"violation","objects":[{"label":"red rectangle","bbox":[111,174,256,319],"confidence":0.99}],"reason":"The image contains a red rectangle, which violates the rule that any red pixel-based rectangular region is considered违规. The red rectangle is present in the original image.","refusal_code":"unsupported_task"}
```

3. 小红矩形

```text
{"result":"violation","objects":[{"label":"red rectangle","bbox":[299,400,414,600]}],"reason":"The image contains a red rectangular region, which is the subject of the violation as per the criteria. The red rectangle is clearly visible and forms the only object in the image.","refusal_code":"unsupported_task"}
```

4. 蓝色矩形

```text
{"result":"violation","objects":[{"label":"red rectangle","bbox":[84,284,240,444],"confidence":0.99}],"reason":"The image contains a blue rectangle, but there is no red rectangle as required by the inspection criteria. Therefore, the result is 'violation' as the red rectangle is not present in the image.","refusal_code":"unsupported_task"}
```

5. 遮挡红色矩形

```text
{"result":"violation","objects":[{"label":"red rectangle","bbox":[77,155,256,824],"confidence":0.98}],"reason":"The image contains a red rectangular region, which is the primary violation as per the inspection criteria. The red rectangle is a red pixel-based shape that is clearly visible and matches the description of a red rectangle.","refusal_code":null}
```

适配器没有把这些失败转换为成功，也没有猜测修复坐标。数值形态疑似模型使用了非原图坐标空间，但原始输出同时存在幻觉、字段缺失和拒答码错误，不能只靠坐标缩放安全修复。

## 命令证据

```bash
uv sync --extra mlx
.venv/bin/hf download mlx-community/Qwen3-VL-2B-Instruct-4bit \
  --revision 9c4f5209e57b31f4b9dfba735de3fb983739c9cc
.venv/bin/python -m src.inference.diagnostics --compact --require-ready
.venv/bin/python -m src.inference.real_probe \
  --config configs/models/qwen3_vl_2b_mlx_4bit.json \
  --image <synthetic-image> --query <criteria> --max-tokens 128 \
  --output <probe-report.json>
```

## 风险与下一实验

1. 在冻结的小型合规集上验证中英文规则表述，区分提示敏感、视觉漏检和规则推理错误。
2. 增加严格 JSON 有限生成或 grammar 约束实验；在此之前保持当前拒绝坏输出的策略。
3. 单独验证 Qwen3-VL 的 grounding 坐标约定。只有官方约定和标注图实验均证明可安全转换时，才考虑显式、可配置的坐标转换；不得对任意越界框猜测缩放。
4. 使用系统级内存采样补充当前进程 RSS，并执行同进程预热后的 20+ 样本延迟实验。
5. 当前模型不得进入质量验收或自动处置路径；继续保留人工复核和 `OUTPUT_VALIDATION_FAILED` 映射。

## 证据状态

| 项目 | 状态 |
|---|---|
| 许可证、revision、下载体积 | recorded |
| 权重哈希 | recorded |
| diagnostics readiness | recorded |
| 单图真实输出 | recorded |
| 5 图真实输出与失败原因 | recorded |
| 真实模型质量验收 | failed |
| 正式数据集精度 | not_run |
| 系统级统一内存峰值 | unavailable |
| 100 请求稳定性 | not_run |
| LoRA/QLoRA 训练 | not_run |
