from __future__ import annotations

import io
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import create_app
from src.core.config import ServiceSettings
from src.core.model_registry import MockModelAdapter, ModelAdapter, build_mock_registry


def image_bytes(image_format: str = "JPEG", size: tuple[int, int] = (32, 24)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(output, format=image_format)
    return output.getvalue()


@pytest.fixture
def settings() -> ServiceSettings:
    return ServiceSettings(
        config_path="test",
        inference_timeout_seconds=0.2,
        concurrency_wait_seconds=0.01,
    )


@pytest.fixture
def client(settings: ServiceSettings) -> TestClient:
    return TestClient(create_app(settings, build_mock_registry()))


@pytest.fixture
def client_factory(settings: ServiceSettings):
    clients: list[TestClient] = []

    def factory(adapter: ModelAdapter | None = None, **setting_overrides: object) -> TestClient:
        overridden = replace(settings, **setting_overrides)
        test_client = TestClient(
            create_app(overridden, build_mock_registry(adapter or MockModelAdapter())),
            raise_server_exceptions=False,
        )
        clients.append(test_client)
        return test_client

    yield factory
    for test_client in clients:
        test_client.close()
