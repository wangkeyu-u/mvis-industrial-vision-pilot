#!/usr/bin/env python3
"""Start the FastAPI service and same-origin demo UI with one command."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGES = (
    "fastapi>=0.115",
    "uvicorn>=0.34",
    "pydantic>=2.10",
    "pyyaml>=6",
    "python-multipart>=0.0.20",
    "pillow>=11",
)


def _current_python_supports_backend() -> bool:
    required = ("fastapi", "uvicorn", "pydantic", "yaml", "multipart", "PIL")
    return all(importlib.util.find_spec(name) is not None for name in required)


def backend_command(app_target: str, host: str, port: int) -> list[str]:
    uvicorn_args = ["-m", "uvicorn", app_target, "--host", host, "--port", str(port)]
    if _current_python_supports_backend():
        try:
            probe = subprocess.run(
                [sys.executable, "-c", "import fastapi, uvicorn, pydantic, yaml, multipart, PIL"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=False,
                timeout=10,
            )
            if probe.returncode == 0:
                return [sys.executable, *uvicorn_args]
        except (OSError, subprocess.TimeoutExpired):
            pass

    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError(
            "FastAPI 运行依赖不完整，且未找到 uv。请先安装 uv，再重试一键启动。"
        )
    command = [uv, "run", "--python", "3.13"]
    for package in RUNTIME_PACKAGES:
        command.extend(("--with", package))
    return [*command, *uvicorn_args]


def ui_command(host: str, port: int, backend_url: str, proxy_timeout: float) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "app" / "server.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--backend-url",
        backend_url,
        "--proxy-timeout",
        str(proxy_timeout),
    ]


def fetch_readiness(url: str, timeout: float = 1.0) -> tuple[bool, dict[str, Any] | None]:
    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
            payload = json.loads(response.read())
            return response.status == 200 and payload.get("status") == "ready", payload
    except HTTPError as error:
        try:
            return False, json.loads(error.read())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False, None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False, None


def wait_until_ready(url: str, process: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"FastAPI 进程提前退出（退出码 {process.returncode}）。")
        ready, payload = fetch_readiness(url)
        last_payload = payload or last_payload
        if ready and payload is not None:
            return payload
        time.sleep(0.1)
    state = last_payload.get("status") if last_payload else "unreachable"
    raise RuntimeError(f"FastAPI readiness 预检超时：{state}")


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键启动视觉合规审查演示。")
    parser.add_argument("--backend-app", default="app.demo_backend:app")
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8011)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8000)
    parser.add_argument("--proxy-timeout", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--open", action="store_true", help="readiness 通过后打开浏览器")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将执行的命令")
    args = parser.parse_args(argv)
    for name in ("backend_port", "ui_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and 65535")
    if args.proxy_timeout <= 0 or args.startup_timeout <= 0:
        parser.error("timeouts must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backend_url = f"http://{args.backend_host}:{args.backend_port}"
    demo_url = f"http://{args.ui_host}:{args.ui_port}/?client=api"
    try:
        backend = backend_command(args.backend_app, args.backend_host, args.backend_port)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        return 1
    frontend = ui_command(args.ui_host, args.ui_port, backend_url, args.proxy_timeout)
    if args.dry_run:
        print("BACKEND", json.dumps(backend, ensure_ascii=False))
        print("UI", json.dumps(frontend, ensure_ascii=False))
        print("DEMO", demo_url)
        return 0

    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_path if existing_path else "")
    backend_process: subprocess.Popen[bytes] | None = None
    ui_process: subprocess.Popen[bytes] | None = None
    try:
        print("① 启动 FastAPI 服务…", flush=True)
        backend_process = subprocess.Popen(backend, cwd=PROJECT_ROOT, env=environment)
        readiness = wait_until_ready(
            f"{backend_url}/health/ready", backend_process, args.startup_timeout
        )
        runtime = readiness.get("details", {}).get("runtime", {})
        print(
            f"✓ readiness 通过：mode={runtime.get('selected_mode', 'unknown')} "
            f"degraded={runtime.get('degraded', 'unknown')}",
            flush=True,
        )
        print("② 启动同源 UI 反向代理…", flush=True)
        ui_process = subprocess.Popen(frontend, cwd=PROJECT_ROOT, env=environment)
        print(f"✓ 演示地址：{demo_url}", flush=True)
        print(f"· JSON 传输：{demo_url}&transport=json", flush=True)
        print(f"· FastAPI 文档：{backend_url}/docs", flush=True)
        print("· 真值声明：当前是真实 FastAPI 链路 + 明确标记的模型替身。", flush=True)
        if args.open:
            webbrowser.open(demo_url)
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: stop_process(ui_process))
        try:
            return ui_process.wait()
        except KeyboardInterrupt:
            print("\n正在停止演示…", flush=True)
            return 0
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    except (OSError, RuntimeError) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        return 1
    finally:
        stop_process(ui_process)
        stop_process(backend_process)


if __name__ == "__main__":
    raise SystemExit(main())
