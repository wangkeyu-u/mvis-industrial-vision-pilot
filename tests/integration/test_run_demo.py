from __future__ import annotations

import io
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from app import run_demo


class RunDemoTest(unittest.TestCase):
    def test_backend_command_uses_current_compatible_python(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with (
            patch.object(run_demo, "_current_python_supports_backend", return_value=True),
            patch.object(run_demo.subprocess, "run", return_value=completed),
        ):
            command = run_demo.backend_command("app.demo_backend:app", "127.0.0.1", 8011)
        self.assertEqual(command[:3], [sys.executable, "-m", "uvicorn"])
        self.assertIn("app.demo_backend:app", command)

    def test_backend_command_falls_back_to_reproducible_uv_environment(self) -> None:
        with (
            patch.object(run_demo, "_current_python_supports_backend", return_value=False),
            patch.object(run_demo.shutil, "which", return_value="/opt/tools/uv"),
        ):
            command = run_demo.backend_command("src.api.app:app", "127.0.0.1", 9001)
        self.assertEqual(command[:5], ["/opt/tools/uv", "run", "--python", "3.13", "--with"])
        self.assertIn("python-multipart>=0.0.20", command)
        self.assertEqual(command[-7:], ["-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", "9001"][-7:])

    def test_wait_until_ready_returns_runtime_evidence(self) -> None:
        process = Mock()
        process.poll.return_value = None
        payload = {"status": "ready", "details": {"runtime": {"selected_mode": "mock-adapter"}}}
        with patch.object(run_demo, "fetch_readiness", return_value=(True, payload)):
            self.assertEqual(run_demo.wait_until_ready("http://demo/health/ready", process, 1), payload)

    def test_dry_run_prints_both_processes_without_starting_them(self) -> None:
        output = io.StringIO()
        with (
            patch.object(run_demo, "backend_command", return_value=["backend"]),
            patch.object(run_demo, "ui_command", return_value=["ui"]),
            patch("sys.stdout", output),
        ):
            exit_code = run_demo.main(["--dry-run", "--ui-port", "8123"])
        self.assertEqual(exit_code, 0)
        rendered = output.getvalue()
        self.assertIn('BACKEND ["backend"]', rendered)
        self.assertIn('UI ["ui"]', rendered)
        self.assertIn("http://127.0.0.1:8123/?client=api", rendered)


if __name__ == "__main__":
    unittest.main()
