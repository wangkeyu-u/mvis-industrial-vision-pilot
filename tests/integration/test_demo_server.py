from __future__ import annotations

import base64
import json
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from app.server import create_server

PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nmock-pixels").decode("ascii")


class DemoServerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = create_server(port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.host, cls.port = cls.server.server_address

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict | str]:
        connection = HTTPConnection(self.host, self.port, timeout=2)
        encoded = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if encoded else {}
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        content_type = response.getheader("Content-Type") or ""
        connection.close()
        payload = json.loads(raw) if "application/json" in content_type else raw.decode("utf-8")
        return response.status, payload

    def valid_request(self, query: str = "找出不符合安全要求的区域") -> dict:
        return {
            "image": PNG_DATA_URL,
            "image_width": 1000,
            "image_height": 600,
            "query": query,
            "task": "inspect",
            "model": "active",
            "use_specialist": True,
        }

    def test_health_version_and_static_ui(self) -> None:
        status, live = self.request("GET", "/health/live")
        self.assertEqual(status, 200)
        self.assertEqual(live["status"], "ok")

        status, version = self.request("GET", "/version")
        self.assertEqual(status, 200)
        self.assertEqual(version["schema_version"], "1.0")

        status, html = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("视证台", html)
        self.assertIn("/ui/app.mjs", html)

    def test_analyze_returns_contract_and_original_pixel_bbox(self) -> None:
        status, payload = self.request("POST", "/v1/analyze", self.valid_request())
        self.assertEqual(status, 200)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["result"], "violation")
        self.assertEqual(payload["objects"][0]["bbox"], [560, 108, 820, 432])
        self.assertEqual(payload["model"]["alias"], "active")
        self.assertFalse(payload["trace"]["image_persisted"])
        self.assertGreater(payload["latency_ms"], 0)
        self.assertTrue(payload["request_id"].startswith("req_demo_"))

    def test_uncertain_refused_and_compliant_have_no_boxes(self) -> None:
        cases = {
            "画面模糊且证据不足": "uncertain",
            "对身份证进行高风险自动判定": "refused",
            "这是合规样例，请确认未发现违规": "compliant",
        }
        for query, expected in cases.items():
            with self.subTest(expected=expected):
                status, payload = self.request("POST", "/v1/analyze", self.valid_request(query))
                self.assertEqual(status, 200)
                self.assertEqual(payload["result"], expected)
                self.assertEqual(payload["objects"], [])

    def test_stable_error_codes(self) -> None:
        invalid_query = self.valid_request("")
        status, payload = self.request("POST", "/v1/analyze", invalid_query)
        self.assertEqual((status, payload["error"]["code"]), (400, "INVALID_QUERY"))
        self.assertIn("request_id", payload)

        invalid_image = self.valid_request()
        invalid_image["image"] = "data:image/png;base64," + base64.b64encode(b"not-a-png").decode("ascii")
        status, payload = self.request("POST", "/v1/analyze", invalid_image)
        self.assertEqual((status, payload["error"]["code"]), (400, "INVALID_IMAGE"))

        status, payload = self.request("POST", "/v1/analyze", self.valid_request("错误演示：模拟超时"))
        self.assertEqual((status, payload["error"]["code"]), (504, "INFERENCE_TIMEOUT"))

    def test_static_route_rejects_path_escape(self) -> None:
        status, _ = self.request("GET", "/ui/../app/server.py")
        self.assertEqual(status, 404)

    def test_real_probe_route_serves_only_manifest_allow_list(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "probes").mkdir()
            image = b"\xff\xd8\xfflicensed-probe"
            (root / "probes" / "allowed.jpg").write_bytes(image)
            (root / "secret.jpg").write_bytes(b"secret")
            (root / "probe_manifest.json").write_text(json.dumps({"probes": [{
                "sample_id": "licensed-1",
                "probe_image": "probes/allowed.jpg",
                "license_id": "cc-by-nc-sa-4.0",
                "attribution": "dataset authors",
                "sha256": "0" * 64,
            }]}), encoding="utf-8")
            server = create_server(port=0, probe_root=root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = HTTPConnection(*server.server_address, timeout=2)
                connection.request("GET", "/demo/real-probes/manifest.json")
                response = connection.getresponse()
                manifest = json.loads(response.read())
                self.assertEqual((response.status, manifest["probes"][0]["sample_id"]), (200, "licensed-1"))
                connection.request("GET", "/demo/real-probes/allowed.jpg")
                response = connection.getresponse()
                self.assertEqual((response.status, response.read()), (200, image))
                connection.request("GET", "/demo/real-probes/secret.jpg")
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 404)
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


class RecordingFastAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _write_json(self, status: int, payload: dict, request_id: str) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("X-Request-ID", request_id)
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except BrokenPipeError:
            pass

    def do_GET(self) -> None:  # noqa: N802
        request_id = self.headers.get("X-Request-ID", "req_upstream")
        if self.path == "/health/ready":
            self._write_json(200, {"status": "ready", "request_id": request_id}, request_id)
        else:
            self._write_json(200, {"status": "live", "request_id": request_id}, request_id)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        request_id = self.headers.get("X-Request-ID", "req_upstream")
        self.server.received.append(  # type: ignore[attr-defined]
            {"body": body, "content_type": self.headers.get("Content-Type"), "request_id": request_id}
        )
        if b"slow-proxy" in body:
            time.sleep(0.08)
        if b"unready-proxy" in body:
            self._write_json(
                503,
                {
                    "request_id": request_id,
                    "error": {"code": "MODEL_NOT_READY", "message": "model is not ready", "request_id": request_id},
                },
                request_id,
            )
            return
        self._write_json(
            200,
            {
                "schema_version": "1.0.0",
                "request_id": request_id,
                "model": {"base": "mock-vlm-0", "adapter": "mock-compliance-v0"},
                "result": "violation",
                "objects": [{"label": "target", "bbox": [1, 1, 4, 4], "confidence": 0.5, "source": "mock"}],
                "reason": "recording upstream",
                "uncertain": True,
                "latency_ms": 2,
                "timing": {"preprocess_ms": 1, "inference_ms": 1, "validation_ms": 0},
                "warnings": ["mock_adapter"],
            },
            request_id,
        )


class FastAPIProxyIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), RecordingFastAPIHandler)
        cls.upstream.received = []  # type: ignore[attr-defined]
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()
        upstream_host, upstream_port = cls.upstream.server_address
        cls.proxy = create_server(port=0, backend_url=f"http://{upstream_host}:{upstream_port}")
        cls.proxy_thread = threading.Thread(target=cls.proxy.serve_forever, daemon=True)
        cls.proxy_thread.start()
        cls.proxy_host, cls.proxy_port = cls.proxy.server_address

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.proxy_thread.join(timeout=2)
        cls.upstream_thread.join(timeout=2)

    def proxy_request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict, dict[str, str]]:
        connection = HTTPConnection(self.proxy_host, self.proxy_port, timeout=2)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = json.loads(response.read())
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        connection.close()
        return response.status, payload, response_headers

    def test_json_and_request_id_are_forwarded_without_cors_wildcard(self) -> None:
        request_id = "req_ui_proxy_json"
        body = json.dumps({"query": "inspect"}).encode("utf-8")
        status, payload, headers = self.proxy_request(
            "POST",
            "/v1/analyze",
            body,
            {"Content-Type": "application/json", "X-Request-ID": request_id, "Origin": "http://localhost:3000"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["request_id"], request_id)
        self.assertEqual(headers["x-request-id"], request_id)
        self.assertEqual(headers["x-mvis-backend"], "fastapi-proxy")
        self.assertEqual(headers["access-control-allow-origin"], "http://localhost:3000")
        received = self.upstream.received[-1]  # type: ignore[attr-defined]
        self.assertEqual(received["body"], body)
        self.assertEqual(received["content_type"], "application/json")
        self.assertEqual(received["request_id"], request_id)

    def test_multipart_content_type_and_body_are_forwarded_unchanged(self) -> None:
        boundary = "----mvis-test-boundary"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"query\"\r\n\r\ninspect\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        status, _, _ = self.proxy_request(
            "POST",
            "/v1/analyze",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        self.assertEqual(status, 200)
        received = self.upstream.received[-1]  # type: ignore[attr-defined]
        self.assertEqual(received["body"], body)
        self.assertEqual(received["content_type"], f"multipart/form-data; boundary={boundary}")

    def test_health_and_structured_not_ready_error_pass_through(self) -> None:
        status, ready, headers = self.proxy_request("GET", "/health/ready")
        self.assertEqual((status, ready["status"]), (200, "ready"))
        self.assertEqual(headers["x-mvis-backend"], "fastapi-proxy")

        body = json.dumps({"query": "unready-proxy"}).encode("utf-8")
        status, payload, headers = self.proxy_request(
            "POST", "/v1/analyze", body, {"Content-Type": "application/json", "X-Request-ID": "req_not_ready"}
        )
        self.assertEqual((status, payload["error"]["code"]), (503, "MODEL_NOT_READY"))
        self.assertEqual(payload["request_id"], headers["x-request-id"])

    def test_preflight_allows_only_local_browser_origins(self) -> None:
        connection = HTTPConnection(self.proxy_host, self.proxy_port, timeout=2)
        connection.request(
            "OPTIONS",
            "/v1/analyze",
            headers={"Origin": "http://127.0.0.1:8000", "Access-Control-Request-Method": "POST"},
        )
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 204)
        self.assertEqual(response.getheader("Access-Control-Allow-Origin"), "http://127.0.0.1:8000")
        connection.close()

    def test_proxy_timeout_and_unreachable_backend_map_to_stable_errors(self) -> None:
        slow_proxy = create_server(
            port=0,
            backend_url=f"http://{self.upstream.server_address[0]}:{self.upstream.server_address[1]}",
            proxy_timeout=0.01,
        )
        slow_thread = threading.Thread(target=slow_proxy.serve_forever, daemon=True)
        slow_thread.start()
        try:
            connection = HTTPConnection(*slow_proxy.server_address, timeout=2)
            body = json.dumps({"query": "slow-proxy"}).encode("utf-8")
            connection.request("POST", "/v1/analyze", body=body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual((response.status, payload["error"]["code"]), (504, "INFERENCE_TIMEOUT"))
            connection.close()
        finally:
            slow_proxy.shutdown()
            slow_proxy.server_close()
            slow_thread.join(timeout=2)

        offline_proxy = create_server(port=0, backend_url="http://127.0.0.1:9")
        offline_thread = threading.Thread(target=offline_proxy.serve_forever, daemon=True)
        offline_thread.start()
        try:
            connection = HTTPConnection(*offline_proxy.server_address, timeout=2)
            body = json.dumps({"query": "inspect"}).encode("utf-8")
            connection.request("POST", "/v1/analyze", body=body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual((response.status, payload["error"]["code"]), (503, "MODEL_NOT_READY"))
            connection.close()
        finally:
            offline_proxy.shutdown()
            offline_proxy.server_close()
            offline_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
