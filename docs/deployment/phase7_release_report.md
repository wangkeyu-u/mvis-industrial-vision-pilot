# Phase 7 最终集成与发布报告

生成时间：`2026-08-09T00:42:53.746931+00:00`

## 验收结论

- Pilot 真实运行验收：`passed`。
- Production 发布：`blocked`。
- 验收模式：`fused`，真实 VLM 与真实 PatchCore specialist 同时运行；Mock 替代：`false`。
- 连续请求：`30/30` 成功；5 次热身不计入延迟统计。
- P50 / P95：`1784.178 / 1883.365 ms`；均值 `1803.463 ms`，范围 `1760.249–1886.032 ms`。
- 进程历史峰值内存：`2730.05 MB`，低于 `12288 MB` 服务预算。
- 受控超时返回 `504 INFERENCE_TIMEOUT`，下一次真实 fused 请求恢复为 `200`。

这只证明当前 M5 MacBook Air 16GB 上的本地 Uvicorn pilot 运行链路可用，不代表模型质量或生产资格通过。

## 最终候选与质量证据

三条路线在同一冻结 56 图 KSDD test 上的算法结果如下。该 test 已参与候选选择，因此属于选择证据，不是独立最终验收。

| 路线 | Macro-F1 | 定位 Acc@IoU 0.5 | P50 / P95 | 发布判断 |
|---|---:|---:|---:|---|
| 零样本 Qwen3-VL | `0.0` | `0.0` | `1882 / 2308 ms` | 拒答过多，不接受 |
| tile QLoRA 384 rank 4 | `0.45631` | `0.0` | `3030 / 4127 ms` | 9 个正例全漏，不接受 |
| PatchCore ResNet-18 | `0.78125` | `0/9` | `7.312 / 15.145 ms` | 仅作为 `pilot_candidate` |

PatchCore 分类正确 `50/56`，困难负例 FPR 为 `2/47`；但定位没有一个样本达到 IoU 0.5。因此后端保留 `quality_status=pilot_failed` 和 `quality_accepted=false`，其 bbox/heatmap 只能用于人工复核，不能用于生产自动决策。

真实运行身份：

- VLM：`mlx-community/Qwen3-VL-2B-Instruct-4bit`；revision `9c4f5209e57b31f4b9dfba735de3fb983739c9cc`；weight hash `4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb`。
- Specialist：`anomalib/patchcore-resnet18-ksdd-v0`；checkpoint revision/weight hash `1ad6587c26805639d9dc6c323a468f0da113f48e2d8e481ae5d0ab595734a681`。
- Specialist manifest/config fingerprint：`67e116edf77de8b1ede6907416dc8abb932afe5f4fe95b857d20c794ba98158e`。
- 数据版本：`ksdd-0.1.0`；VLM prompt 版本：`ksdd_prompt_v1`。
- QLoRA 候选 adapter hash：`69b4ff0ea1ee7b313ef1276d762c92c13cd02b8e6c01095237b8bf7848eb9275`；只用于真实生命周期/回滚探针，不是最终质量候选。

## API 融合契约

`/v1/analyze` 明确支持 `vlm_only`、`specialist_only`、`fused`：

- specialist 返回 `score`、`threshold`、`source`、specialist-owned `bbox` 和有时限的 PNG heatmap artifact。
- fused 一致时，最终定位仍只采用 specialist bbox。
- VLM 与 specialist 的结论或几何冲突时，响应标记 `uncertain=true`、`human_review_required=true`；保留 specialist 定位证据并丢弃 VLM 框，禁止 VLM 编造最终 bbox。
- 当前质量未接受时，响应、`/version`、`/v1/models` 和结构化日志持续携带 `pilot_failed` 与完整 provenance。

热力图通过 `/v1/artifacts/heatmaps/{artifact_id}` 短期提供；上传内容、热力图和 bbox 均经过大小、格式、像素和边界校验，artifact 响应使用 `no-store`、`nosniff` 和 CSP。

## 真实 Uvicorn 验收协议

- 设备：Apple arm64，物理内存 `16384 MB`，Darwin `25.5.0`，Python `3.13.13`。
- 传输：独立进程 `real_uvicorn_http`；并发 `1`；不允许 Mock 替代。
- 样本：`Part0.jpg`，`500×1265`，`277402` bytes，SHA-256 `86148af3a4c148900096165380e6e88f9c2f0573e04e2b2b309efd751f8fef95`。
- 初始响应：`compliant`；specialist score `11.9370326996`，阈值 `13.0968675613`，返回真实 heatmap artifact。
- 失败恢复：服务边界一次性故障注入得到 `504`；后续真实 fused 推理得到 `200`。故障和恢复请求均排除在 30 次延迟统计之外。

执行命令：

```bash
MVIS_MODEL_MODE=real \
MVIS_ANALYSIS_MODE=fused \
MVIS_SPECIALIST_MANIFEST=artifacts/model/phase7/patchcore_resnet18/run_manifest.json \
MVIS_SPECIALIST_QUALITY_STATUS=pilot_failed \
docs/deployment/mvis.sh release-validate \
  fused data/processed/ksdd_v0/images/kos10/Part0.jpg
```

机器可读原始证据为 `phase7_release_report.json`。

## 回滚和失败安全

主 30 请求验收启动时 active 为 zero-shot，registry 中没有可回滚的 ready previous，所以主报告诚实记录 `no_ready_previous_real_candidate`，没有伪造一次回滚。

随后在隔离的真实 registry 中加载本地 QLoRA 候选并激活为 pilot，创建 `previous=zero-shot`，实际执行原子回滚：

- from：`qwen3-vl-2b-instruct-4bit-lora-r4-tile384`
- to：`qwen3-vl-2b-instruct-4bit`
- 审计 action/result：`rollback / success`
- 回滚后真实 `specialist_only` HTTP 探针：`200`
- Mock 替代：`false`

完整证据见 `phase7_real_rollback_probe.md` 和 `phase7_real_rollback_probe.json`。该探针只证明 lifecycle、审计和恢复链路，不给 QLoRA 候选增加任何质量资格。

## Production 门禁与剩余风险

Production active 必须同时满足：runtime ready、`quality_status=pilot_passed`、精确绑定 model/revision/adapter/data/prompt provenance 的 evaluator 报告，以及验签成功。目前 evaluator attestation 缺失，且定位指标失败，所以 `production_ready=false`、`production_release_status=blocked`。

发布前仍需：

1. 使用未参与候选选择的新冻结 holdout 复验分类，消除 selection bias。
2. 重新设计并验证定位提取，在签名评测中通过定位门禁。
3. 完成 KSDD `CC-BY-NC-SA-4.0` 与 ImageNet 预训练权重的商业许可审查。
4. 将当前进程内 ModelOps 审计链送入不可变外部日志存储。

## 工件与回归

- 机器报告：`docs/deployment/phase7_release_report.json`
- 回滚证据：`docs/deployment/phase7_real_rollback_probe.json`
- 算法报告：`docs/model/phase7_final_model_report.md`
- Specialist manifest：`artifacts/model/phase7/patchcore_resnet18/run_manifest.json`
- 统一入口：`docs/deployment/mvis.sh`

最终回归（同一工作树）：

- API + VLM bridge + specialist bridge：`95 passed`，`44 warnings`。
- 统一入口 `docs/deployment/mvis.sh test`：Ruff 通过，API `85 passed`。
- 全仓：`246 passed`，`12 subtests passed`，`44 warnings`。
- Ruff：`All checks passed!`。
- `git diff --check`：通过。

warning 为 1 个 Starlette TestClient 迁移提示及 43 个 Torch JIT deprecation；没有测试失败。
