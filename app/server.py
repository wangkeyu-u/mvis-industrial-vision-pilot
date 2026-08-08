#!/usr/bin/env python3
"""Dependency-free UI server and optional same-origin FastAPI reverse proxy.

Without ``--backend-url`` the server keeps the deterministic contract mock for
offline demos. With a backend URL, API/health/version requests are forwarded
byte-for-byte to the real FastAPI service so browsers do not require permissive
CORS settings. Uploaded images are never persisted by this process.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import mimetypes
import re
import secrets
import socket
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = PROJECT_ROOT / "ui"
CLIENT_ROOT = PROJECT_ROOT / "src" / "client"
ASSET_ROOT = PROJECT_ROOT / "assets"
DEFAULT_PROBE_ROOT = PROJECT_ROOT / "data" / "processed" / "ksdd_v0"
MAX_BODY_BYTES = 14 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
PROXY_PATHS = {"/v1/analyze", "/health/live", "/health/ready", "/version", "/v1/models"}


@dataclass(frozen=True)
class Route:
    prefix: str
    root: Path


STATIC_ROUTES = (
    Route("/ui/", UI_ROOT),
    Route("/src/client/", CLIENT_ROOT),
    Route("/assets/", ASSET_ROOT),
)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _request_id() -> str:
    return f"req_demo_{int(time.time() * 1000)}_{secrets.token_hex(2)}"


def _error(code: str, message: str, request_id: str, *, details: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"request_id": request_id, "error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return payload


def _decode_data_url(value: object) -> tuple[str, bytes]:
    if not isinstance(value, str):
        raise ValueError("图片必须使用 base64 data URL。")
    matched = re.fullmatch(r"data:([^;,]+);base64,(.+)", value, flags=re.DOTALL)
    if not matched:
        raise ValueError("图片 data URL 格式无效。")
    mime_type, encoded = matched.groups()
    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError("仅支持 JPG、PNG、WEBP。")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("图片 base64 内容无效。") from exc
    if not raw:
        raise ValueError("图片内容为空。")
    signatures = {
        "image/jpeg": (b"\xff\xd8\xff",),
        "image/png": (b"\x89PNG\r\n\x1a\n",),
        "image/webp": (b"RIFF",),
    }
    if not any(raw.startswith(signature) for signature in signatures[mime_type]):
        raise ValueError("图片 MIME 与文件签名不一致。")
    if mime_type == "image/webp" and (len(raw) < 12 or raw[8:12] != b"WEBP"):
        raise ValueError("WEBP 文件签名无效。")
    return mime_type, raw


def build_mock_analysis(payload: dict[str, Any], request_id: str) -> dict[str, Any]:
    """Build a deterministic response that follows the frozen `/v1` contract."""
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip() or len(query.strip()) > 1000:
        raise LookupError("INVALID_QUERY")
    _decode_data_url(payload.get("image"))

    lowered = query.strip().lower()
    width = max(1, int(payload.get("image_width") or 1280))
    height = max(1, int(payload.get("image_height") or 800))
    latency_ms = 386
    result = "violation"
    uncertain = False
    objects: list[dict[str, Any]] = [
        {
            "label": "未佩戴防护装备",
            "bbox": [round(width * 0.56), round(height * 0.18), round(width * 0.82), round(height * 0.72)],
            "confidence": 0.91,
            "source": "vlm",
        }
    ]
    reason = "检测到人员作业区域存在防护装备缺失，请人工复核标注区域。"
    warnings = ["当前为可替换的演示 Mock，结果不得用于业务决策。"]

    if any(token in lowered for token in ("timeout", "超时", "错误演示")):
        raise TimeoutError("INFERENCE_TIMEOUT")
    if any(token in lowered for token in ("拒答", "高风险", "身份证", "人脸识别", "medical")):
        result = "refused"
        uncertain = True
        objects = []
        reason = "该请求属于高风险或超出系统用途范围，已拒绝自动判断，请交由人工处理。"
        warnings.append("系统不会对高风险场景给出确定性结论。")
    elif any(token in lowered for token in ("不确定", "证据不足", "遮挡", "uncertain", "模糊")):
        result = "uncertain"
        uncertain = True
        objects = []
        reason = "图像证据不足，无法形成可靠结论；建议补充更清晰、无遮挡的图片。"
        warnings.append("置信度低于策略阈值，未生成边界框。")
    elif any(token in lowered for token in ("未发现", "通过", "compliant", "合规样例")):
        result = "compliant"
        objects = []
        reason = "在当前图片与查询范围内未发现明确违规证据。"

    return {
        "schema_version": "1.0",
        "request_id": request_id,
        "model": {
            "base": "qwen3-vl-2b-instruct-4bit",
            "adapter": "compliance-lora-v0.1-demo",
            "alias": payload.get("model") or "active",
            "quantization": "4-bit",
        },
        "result": result,
        "objects": objects,
        "reason": reason,
        "uncertain": uncertain,
        "latency_ms": latency_ms,
        "timing": {"preprocess_ms": 42, "inference_ms": 311, "validation_ms": 33},
        "warnings": warnings,
        "trace": {
            "task": payload.get("task") or "inspect",
            "use_specialist": bool(payload.get("use_specialist", True)),
            "source": "mock-api",
            "image_persisted": False,
        },
    }


class DemoRequestHandler(BaseHTTPRequestHandler):
    server_version = "MVISDemo/0.1"

    def log_message(self, format: str, *args: object) -> None:
        # Keep the local demo readable while avoiding query or image logging.
        print(f"[demo] {self.address_string()} {format % args}")

    def _local_cors_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        if not origin:
            return None
        parsed = urlsplit(origin)
        return origin if parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost", "::1"} else None

    def _common_headers(self) -> None:
        origin = self._local_cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _write_body(self, body: bytes) -> None:
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Browser cancellation closes the downstream socket. The upstream
            # connection is closed by the proxy context immediately afterwards.
            return

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._common_headers()
        self.end_headers()
        self._write_body(body)

    def _send_file(self, path: Path) -> None:
        body = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self._common_headers()
        self.end_headers()
        self._write_body(body)

    def _probe_manifest(self) -> dict[str, Any] | None:
        manifest_path = self._proxy_server.probe_root / "probe_manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        probes = payload.get("probes") if isinstance(payload, dict) else None
        return payload if isinstance(probes, list) else None

    @property
    def _proxy_server(self) -> "DemoHTTPServer":
        return self.server  # type: ignore[return-value]

    def _proxy_request(self, method: str, path: str, body: bytes | None = None) -> None:
        backend_url = self._proxy_server.backend_url
        if not backend_url:
            raise RuntimeError("proxy requested without backend URL")
        parsed = urlsplit(backend_url)
        connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        base_path = parsed.path.rstrip("/")
        target = f"{base_path}{path}"
        request_id = self.headers.get("X-Request-ID", "")
        if not REQUEST_ID_PATTERN.fullmatch(request_id):
            request_id = _request_id()
        headers = {
            "Accept": self.headers.get("Accept", "application/json"),
            "X-Request-ID": request_id,
            "X-Forwarded-Host": self.headers.get("Host", ""),
            "X-Forwarded-Proto": "http",
        }
        if body is not None:
            headers["Content-Type"] = self.headers.get("Content-Type", "application/octet-stream")
            headers["Content-Length"] = str(len(body))
        connection = connection_class(parsed.hostname, port, timeout=self._proxy_server.proxy_timeout)
        try:
            connection.request(method, target, body=body, headers=headers)
            upstream = connection.getresponse()
            response_body = upstream.read(MAX_BODY_BYTES + 1)
            if len(response_body) > MAX_BODY_BYTES:
                raise ValueError("FastAPI response exceeds proxy limit")
            self.send_response(upstream.status)
            self.send_header("Content-Type", upstream.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Request-ID", upstream.getheader("X-Request-ID", request_id))
            self.send_header("X-MVIS-Backend", "fastapi-proxy")
            self._common_headers()
            self.end_headers()
            self._write_body(response_body)
        except (TimeoutError, socket.timeout):
            self._send_json(
                _error("INFERENCE_TIMEOUT", "FastAPI 服务响应超时，请取消后重试。", request_id),
                HTTPStatus.GATEWAY_TIMEOUT,
            )
        except (ConnectionError, OSError, http.client.HTTPException):
            self._send_json(
                _error("MODEL_NOT_READY", "无法连接 FastAPI 服务；请确认后端已启动且模型就绪。", request_id),
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except ValueError as exc:
            self._send_json(_error("INTERNAL_ERROR", str(exc), request_id), HTTPStatus.BAD_GATEWAY)
        finally:
            connection.close()

    @staticmethod
    def _safe_path(root: Path, relative: str) -> Path | None:
        candidate = (root / unquote(relative)).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(UI_ROOT / "index.html")
            return
        if path == "/demo/health/live":
            self._send_json({"status": "ok", "service": "mvis-demo"})
            return
        if path == "/demo/real-probes/manifest.json":
            manifest = self._probe_manifest()
            if manifest is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self._send_json(manifest)
            return
        if path.startswith("/demo/real-probes/"):
            manifest = self._probe_manifest()
            requested_name = unquote(path.removeprefix("/demo/real-probes/"))
            allowed = {
                Path(str(item.get("probe_image", ""))).name: str(item.get("probe_image", ""))
                for item in (manifest or {}).get("probes", [])
                if isinstance(item, dict) and item.get("probe_image")
            }
            relative = allowed.get(requested_name) if "/" not in requested_name and "\\" not in requested_name else None
            resolved = self._safe_path(self._proxy_server.probe_root, relative) if relative else None
            if resolved:
                self._send_file(resolved)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        if path in PROXY_PATHS and self._proxy_server.backend_url:
            self._proxy_request("GET", path)
            return
        if path == "/health/live":
            self._send_json({"status": "ok", "service": "mvis-demo"})
            return
        if path == "/health/ready":
            self._send_json({"status": "ready", "backend": "mock", "model_ready": False})
            return
        if path == "/version":
            self._send_json({"service": "mvis-demo", "version": "0.1.0", "schema_version": "1.0"})
            return
        for route in STATIC_ROUTES:
            if path.startswith(route.prefix):
                resolved = self._safe_path(route.root, path.removeprefix(route.prefix))
                if resolved:
                    self._send_file(resolved)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path not in PROXY_PATHS:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Request-ID")
        self.send_header("Access-Control-Max-Age", "600")
        self._common_headers()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != "/v1/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        request_id = _request_id()
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(
                _error("INVALID_IMAGE", "请求体为空或超过演示服务限制。", request_id),
                HTTPStatus.BAD_REQUEST,
            )
            return
        body = self.rfile.read(length)
        if self._proxy_server.backend_url:
            self._proxy_request("POST", "/v1/analyze", body)
            return
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON object。")
            result = build_mock_analysis(payload, request_id)
        except json.JSONDecodeError:
            self._send_json(_error("INVALID_IMAGE", "请求 JSON 无效。", request_id), HTTPStatus.BAD_REQUEST)
        except LookupError:
            self._send_json(_error("INVALID_QUERY", "查询必须为 1–1000 个字符。", request_id), HTTPStatus.BAD_REQUEST)
        except ValueError as exc:
            self._send_json(_error("INVALID_IMAGE", str(exc), request_id), HTTPStatus.BAD_REQUEST)
        except TimeoutError:
            self._send_json(
                _error("INFERENCE_TIMEOUT", "模拟推理超时；请重试或降低输入复杂度。", request_id),
                HTTPStatus.GATEWAY_TIMEOUT,
            )
        else:
            self._send_json(result)


class DemoHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        backend_url: str | None = None,
        proxy_timeout: float = 10.0,
        probe_root: Path = DEFAULT_PROBE_ROOT,
    ) -> None:
        self.backend_url = backend_url.rstrip("/") if backend_url else None
        self.proxy_timeout = proxy_timeout
        self.probe_root = probe_root.resolve()
        super().__init__(server_address, DemoRequestHandler)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    backend_url: str | None = None,
    proxy_timeout: float = 10.0,
    probe_root: Path = DEFAULT_PROBE_ROOT,
) -> DemoHTTPServer:
    return DemoHTTPServer(
        (host, port), backend_url=backend_url, proxy_timeout=proxy_timeout, probe_root=probe_root
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local visual compliance demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-url", help="FastAPI base URL; enables same-origin proxy mode")
    parser.add_argument("--proxy-timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.proxy_timeout <= 0:
        parser.error("--proxy-timeout must be positive")
    server = create_server(
        args.host,
        args.port,
        backend_url=args.backend_url,
        proxy_timeout=args.proxy_timeout,
    )
    host, port = server.server_address
    print(f"视觉合规审查演示已启动：http://{host}:{port}")
    if server.backend_url:
        print(f"FastAPI 同源代理：{server.backend_url}")
        print("访问 ?client=api 使用 multipart；添加 &transport=json 切换 base64 JSON。")
    else:
        print("离线模式：浏览器 Mock 可用；?client=api 指向本地契约 Mock。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n演示服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
