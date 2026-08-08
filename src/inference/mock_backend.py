"""Deterministic no-weight backend used for integration tests and local smoke runs."""

from __future__ import annotations

import hashlib
import json
from typing import Callable

from .contracts import BackendRequest, BackendResponse

ResponseFactory = Callable[[BackendRequest], str]


class MockBackend:
    def __init__(self, response_factory: ResponseFactory | None = None) -> None:
        self._response_factory = response_factory or self._default_response
        self.calls: list[BackendRequest] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def ready(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.calls.append(request)
        text = self._response_factory(request)
        digest = hashlib.sha256(
            f"{request.generation.seed}:{request.prompt}".encode("utf-8")
        ).hexdigest()[:12]
        return BackendResponse(text=text, metadata={"mock_digest": digest})

    @staticmethod
    def _default_response(request: BackendRequest) -> str:
        prompt = request.prompt.lower()
        start = prompt.rfind("<criteria>")
        end = prompt.rfind("</criteria>")
        criteria = prompt[start + len("<criteria>") : end] if start >= 0 and end > start else prompt
        refusal_markers = ("证据不足", "看不清", "insufficient", "ambiguous")
        if any(token in criteria for token in refusal_markers):
            payload = {
                "result": "uncertain",
                "objects": [],
                "reason": "可见证据不足，需人工复核。",
                "refusal_code": "insufficient_evidence",
            }
        elif any(token in criteria for token in ("违规", "violation", "non-compliant")):
            payload = {
                "result": "violation",
                "objects": [
                    {"label": "target", "bbox": [10, 12, 40, 44], "confidence": 0.9}
                ],
                "reason": "检测到与检查条件匹配的可见区域。",
                "refusal_code": None,
            }
        else:
            payload = {
                "result": "compliant",
                "objects": [],
                "reason": "未检测到与检查条件匹配的违规证据。",
                "refusal_code": None,
            }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
