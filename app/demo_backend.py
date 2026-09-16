"""Deterministic FastAPI adapter used by the local demo.

This module deliberately exercises the production FastAPI request, image
validation, cancellation, timeout, error and response-schema paths while the
model boundary is replaced by an explicitly labelled test double.  It is not
model-quality evidence and must never be presented as such.
"""

from __future__ import annotations

import asyncio

from src.api.app import create_app
from src.core.model_registry import AdapterRequest, build_mock_registry
from src.core.schemas import EvidenceObject, ModelIdentity, ModelOutput, ObjectSource


class DemoStateAdapter:
    """Stable state adapter for success, uncertainty, refusal and cancellation."""

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            base="fastapi-demo-test-double",
            adapter="visual-compliance-states-v1",
        )

    @property
    def ready(self) -> bool:
        return True

    async def analyze(self, request: AdapterRequest) -> ModelOutput:
        query = request.query.casefold()
        if any(token in query for token in ("slow", "取消", "超时")):
            # Long enough for a human/browser to exercise AbortController.
            await asyncio.sleep(2.0)

        if any(token in query for token in ("身份证", "人脸识别", "高风险", "refuse", "拒答")):
            return ModelOutput(
                result="uncertain",
                objects=[],
                reason="该请求属于高风险或超出演示系统用途，已拒绝自动判断。",
                uncertain=True,
                warnings=[
                    "demo_adapter",
                    "refusal:high_risk_or_out_of_scope",
                    "not_model_quality_evidence",
                ],
            )

        if any(token in query for token in ("不确定", "证据不足", "模糊", "遮挡", "uncertain")):
            return ModelOutput(
                result="uncertain",
                objects=[],
                reason="图像证据低于复核阈值，无法形成可靠结论。",
                uncertain=True,
                warnings=["demo_adapter", "low_confidence", "not_model_quality_evidence"],
            )

        if any(token in query for token in ("合规样例", "未发现", "compliant", "确认合规")):
            return ModelOutput(
                result="compliant",
                objects=[],
                reason="演示适配器在当前查询范围内未返回明确违规证据。",
                uncertain=False,
                warnings=["demo_adapter", "not_model_quality_evidence"],
            )

        width, height = request.image.width, request.image.height
        return ModelOutput(
            result="violation",
            objects=[
                EvidenceObject(
                    label="疑似不合规区域（演示）",
                    bbox=(
                        round(width * 0.22, 2),
                        round(height * 0.18, 2),
                        round(width * 0.78, 2),
                        round(height * 0.76, 2),
                    ),
                    confidence=0.82,
                    source=ObjectSource.MOCK,
                )
            ],
            reason="演示适配器返回了稳定证据框，用于验证 FastAPI 契约和 UI 渲染链路。",
            uncertain=False,
            warnings=["demo_adapter", "not_model_quality_evidence"],
        )


registry = build_mock_registry(DemoStateAdapter())
registry.set_runtime_status(
    requested_mode="demo",
    selected_mode="mock-adapter",
    degraded=True,
    fallback_reason="deterministic local demo; no production model loaded",
)

app = create_app(registry=registry)
