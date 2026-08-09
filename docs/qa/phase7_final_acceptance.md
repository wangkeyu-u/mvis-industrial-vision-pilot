# 第七阶段最终面试验收报告

执行日期：2026-08-09
目标设备：Apple Silicon / 16 GB unified memory
结论：**产品演示、真实评测包对比和真实服务阻断证据可运行；specialist/fused 真实在线定位仍阻塞。**

这不是模型正式验收通过。当前 56 张 KSDD 冻结测试集只能用于 `pilot_only`；PatchCore 虽然将 Macro-F1 从 0.0% 提高到 78.1%，但 Acc@IoU 0.5 仍为 0.0%，困难负例 FPR 从 0.0% 回退到 4.3%，所以状态为 `PILOT_FAILED / FORMAL KPI LOCKED`。

## 本阶段交付

- 主界面支持 `vlm_only`、`specialist_only`和 `fused`，请求会显式携带 `analysis_mode`。
- 协作证据链分开展示 specialist 来源/revision/框数/热力图、VLM 解释、fusion 冲突、不确定和人工复核原因。
- 真实定位标记 fail closed：`vlm_only`永不会获得真实定位；`specialist_only`必须绑定 real/non-degraded readiness 和真实 specialist 来源；`fused`还必须有 `fused_evidence_agreement`且无冲突/人工复核。
- 真实评测双包支持 specialist 候选人；数据 manifest 或 sample ID 不同时不渲染对比。LoRA/fused 候选另强制 prompt、基础 revision 和 adapter hash。
- readiness 现在分别呈现 active VLM、specialist 和三种 analysis mode 的 `runtime_ready`与 `quality_status`。选中模式未就绪时在发分析请求前阻断。
- 附件截图都来自实际浏览器回归：
  - [真实运行诊断](phase7_real_runtime_diagnostics.jpg)
  - [真实评测包配对](phase7_real_evaluation_evidence.jpg)
  - [真实 specialist/fused 阻断](phase7_real_specialist_blocked.jpg)
  - [真实 VLM 输出校验失败](phase7_real_vlm_failure.jpg)

## 真实服务证据

启动时使用实际 MLX VLM 与 Phase 7 PatchCore manifest：

```bash
MVIS_MODEL_MODE=real \
MVIS_SPECIALIST_MANIFEST="$PWD/artifacts/model/phase7/patchcore_resnet18/run_manifest.json" \
.venv/bin/python -m uvicorn src.api.app:app --host 127.0.0.1 --port 19710

.venv/bin/python app/server.py --host 127.0.0.1 --port 19700 \
  --backend-url http://127.0.0.1:19710 --proxy-timeout 15
```

浏览器入口：`http://127.0.0.1:19700/?client=api&acceptance=real`。

| 字段 | 实测值 |
|---|---|
| 服务启动 readiness request_id | `req_b706ea639e5b44aea65b1b47a6e2164a` |
| fused 请求前 readiness request_id | `req_ui_1786235504312_6b7de732c63f4d77` |
| VLM-only 请求前 readiness request_id | `req_ui_1786235524026_a3c985bd6a344cbe` |
| requested / selected mode | `real / real` |
| degraded | `false` |
| active VLM revision | `9c4f5209e57b31f4b9dfba735de3fb983739c9cc` |
| VLM quality | `pilot_failed / quality_accepted=false` |
| specialist | `anomalib/patchcore-resnet18-ksdd-v0` |
| specialist revision | `1ad6587c26805639d9dc6c323a468f0da113f48e2d8e481ae5d0ab595734a681` |
| specialist runtime / quality | `runtime_ready=false / unvalidated` |
| analysis modes | `vlm_only=runtime_ready`; `specialist_only/fused=runtime_blocked` |

### 真实请求观察

1. 许可样例 `ksdd_kos10_part3` 选择 `fused`时，UI 根据同一次 readiness 在发请求前返回 `MODEL_NOT_READY`；无旧框、无结果 JSON、无“真实定位”印章。
2. 关闭 specialist 后执行真实 `vlm_only`，FastAPI 在 7797 ms HTTP 总耗时后返回 `422 OUTPUT_VALIDATION_FAILED`，分析 request_id 为 `req_ui_1786235524032_c185f0eaf63543ca`。页面仍不保留旧证据。
3. 因为未获得可验收的 specialist/fused 成功响应，本报告没有伪造真实定位框或热力图截图。

## 真实评测包浏览器验收

每个槽位一次选中评测包的 7 个 JSON 组件，再附加同一运行的 `run_manifest.json`。浏览器重新计算 package manifest 中每个 JSON 的 SHA-256。

| 证据 | VLM 零样本 | PatchCore specialist |
|---|---:|---:|
| package 真值 | verified, non-mock, non-fixture | verified, non-mock, non-fixture |
| dataset | `ksdd-0.1.0` | `ksdd-0.1.0` |
| manifest SHA-256 | `fe9983728d7aa830f11a1369ff1a4eb353be8e8403685a1cf45ea23db128085a` | 相同 |
| sample IDs | 56 | 56，exact match |
| Macro-F1 | 0.0% | 78.1% (95% CI 59.6–91.8%) |
| Acc@IoU 0.5 | 0.0% | 0.0% |
| JSON 有效率 | 98.2% | 100.0% |
| 困难负例 FPR | 0.0% | 4.3% (95% CI 0.0–10.6%) |
| 证据—结论一致率 | 98.2% | 100.0% |
| P50 / P95 | 1882 / 2308 ms | 7.3 / 15.1 ms |
| process RSS peak | 707 MB | 1915 MB |
| MLX allocator peak | 2763 MB | not recorded |
| 最终质量状态 | `pilot_failed` | `pilot_failed` |

最大改善在图片级分类和延迟；最大失败是 9 个定位正例的 Acc@IoU 全部为 0，且 47 个困难负例中有 2 个误报。界面因此同时展示提升、回退和失败切片。

## 浏览器回归

| 场景 | 结果 |
|---|---|
| Mock fused | 显示合成热力区、VLM 解释、Mock fusion、人工复核和 `REAL LOCALIZATION BLOCKED` |
| Mock specialist_only | 显示 specialist 来源，不伪造 VLM 解释，真实定位仍阻断 |
| 不确定 / 拒答 / 超时 | 无旧框；超时后 JSON 导出撤销，验收证据只记录本次错误 |
| 390×844 | 模式选择可达，证据链单列，冲突/复核/阻断标记可读 |
| 真实 API fused | readiness 预检阻断，无陈旧结果 |
| 真实 API VLM-only | 422 结构失败，无陈旧结果 |
| 真实双评测包 | 哈希、manifest、sample ID、revision、指标/CI、切片、失败案例和资源可读 |

## 产品/API 契约

```text
GET /health/ready
  -> request_id
  -> runtime {requested_mode, selected_mode, degraded, fallback_reason}
  -> models[], specialists[], analysis_modes{}

POST /v1/analyze
  multipart: image, query, task, model, use_specialist, analysis_mode, options
  JSON: image(data URL), image_width, image_height, query, task, model,
        use_specialist, analysis_mode, options

  analysis_mode: vlm_only | specialist_only | fused
  -> request_id, analysis_mode, model, provenance
  -> quality_status, quality_accepted, serving_tier
  -> specialist? {specialist_id, revision, score, threshold, detected,
                   source, objects, heatmap, provenance, quality_status,
                   quality_accepted}
  -> human_review_required, result, objects, reason, uncertain
  -> latency_ms, latency/timing, warnings

GET /v1/artifacts/heatmaps/{hm_id}
  -> private, no-store image/png; URI 有 TTL，不暴露本地路径
```

fusion 中输出框仍保持 specialist 的 `source`(例如 `patchcore`)。只有 `fused_evidence_agreement`才会获得融合门禁通过；`model_conflict`、`human_review_required=true`或 `uncertain=true`都会阻止真实定位标记。

## 自动化证据

| 层级 | 结果 |
|---|---:|
| 演示服务/同源代理 unittest | 15/15 |
| UI 静态、响应式、可访问性契约 | 8/8 |
| 客户端、评测导入、真实门禁 Node 测试 | 24/24 |
| 真实 FastAPI 网络 E2E | 5/5 |
| 仓库全量 pytest | 242 passed + 12 subtests |
| 前端语法检查 | 6/6 |
| scoped diff whitespace 检查 | 通过 |

全量 pytest 只有 1 条已知上游弃用警告：Starlette TestClient 建议从 `httpx` 迁移到 `httpx2`；不影响本次结果。核心命令：

```bash
python3 -m unittest discover -s tests/integration -p 'test_*.py' -v
python3 -m unittest discover -s tests/ui -p 'test_*.py' -v
node --test tests/ui/*.test.mjs
uv run --python 3.13 pytest -q tests/integration/fastapi_proxy_e2e_test.py
uv run --python 3.13 pytest -q
node --check ui/app.mjs ui/evaluation-panel.mjs \
  src/client/api-client.mjs src/client/evaluation-report.mjs \
  src/client/real-acceptance.mjs src/client/evidence-presentation.mjs
git diff --check -- ui src/client docs/qa
```

## 已知限制与发布判断

1. **specialist 运行阻塞**：artifact/manifest 已注册，但 service bridge 尚未将 PatchCore checkpoint 加载成 ready adapter。
2. **真实 fused 未观察**：没有成功的真实 specialist/fused 响应，所以没有定位框、heatmap URI 或 fused request_id 截图。
3. **定位质量未达标**：PatchCore 的 Acc@IoU 0.5 为 0，不能把图片级分类改善解读为可验收定位。
4. **pilot 样本不足**：56 张小于正式 KPI 门槛 300；所有数字只是 pilot。
5. **LoRA 仍无真实 adapter**：M5/16GB 上的训练 smoke 超资源预算，本阶段选用实际 specialist 包作为候选对比，未冒充 LoRA。
6. **热力图同源代理待闭环**：FastAPI 已有有界、短期的 heatmap 路由；当前 `app/server.py` 的 GET 代理 allowlist 尚未包含动态 heatmap 路径。

发布判断：**可用于面试演示产品工程、证据治理和失败复盘；不可声称 specialist/fused 真实在线定位或正式模型 KPI 通过。**
