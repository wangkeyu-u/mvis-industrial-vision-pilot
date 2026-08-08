"""Lazy MLX-VLM backend for local Qwen3-VL inference on Apple Silicon."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from .cache import resolve_cached_model_path
from .config import ModelConfig
from .contracts import BackendRequest, BackendResponse
from .errors import ModelNotReadyError


class MlxVlmBackend:
    """Thin runtime wrapper; importing this module never imports or loads MLX."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self._generate: Any = None
        self._apply_chat_template: Any = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        return "mlx_vlm"

    @property
    def ready(self) -> bool:
        return self._model is not None and self._processor is not None

    def _model_source(self) -> str:
        cached_path = resolve_cached_model_path(self.config)
        if cached_path is not None:
            return str(cached_path)
        if not self.config.allow_download:
            raise ModelNotReadyError(
                "model download is disabled and no complete pinned MLX snapshot is cached"
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ModelNotReadyError(
                "huggingface-hub is required when allow_download=true"
            ) from exc
        try:
            return snapshot_download(
                repo_id=self.config.model_id,
                revision=self.config.revision,
            )
        except Exception as exc:
            raise ModelNotReadyError(f"failed to obtain pinned model snapshot: {exc}") from exc

    def load(self) -> None:
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if self.ready:
            return
        if self.config.trust_remote_code:
            raise ModelNotReadyError("trust_remote_code must remain false for registered weights")
        source = self._model_source()
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:
            raise ModelNotReadyError(
                "mlx-vlm is not installed; install the pinned Apple Silicon runtime first"
            ) from exc
        try:
            self._model, self._processor = load(
                source,
                adapter_path=self.config.adapter_path,
                trust_remote_code=False,
            )
        except TypeError:
            # Compatibility with older pinned mlx-vlm versions whose load()
            # does not yet expose adapter_path/trust_remote_code.
            if self.config.adapter_path:
                raise ModelNotReadyError(
                    "installed mlx-vlm does not support adapter_path in load()"
                )
            self._model, self._processor = load(source)
        except Exception as exc:
            raise ModelNotReadyError(f"failed to load local MLX model: {exc}") from exc
        self._generate = generate
        self._apply_chat_template = apply_chat_template

    def generate(self, request: BackendRequest) -> BackendResponse:
        # MLX random state is process-global. Serializing seed + generation is
        # required for repeatability and also bounds unified-memory pressure.
        with self._lock:
            return self._generate_locked(request)

    def _generate_locked(self, request: BackendRequest) -> BackendResponse:
        if not self.ready:
            self._load_locked()
        if not isinstance(request.image, (str, Path)):
            raise ModelNotReadyError("MLX backend currently requires a local image path")
        image_path = Path(request.image).expanduser()
        if not image_path.is_file():
            raise ModelNotReadyError(f"local image path does not exist: {image_path}")

        try:
            import mlx.core as mx

            mx.random.seed(request.generation.seed)
            formatted = self._apply_chat_template(
                self._processor,
                self._model.config,
                request.prompt,
                num_images=1,
            )
            output = self._generate(
                self._model,
                self._processor,
                formatted,
                [str(image_path)],
                max_tokens=request.generation.max_tokens,
                temperature=request.generation.temperature,
                top_p=request.generation.top_p,
                verbose=False,
            )
        except Exception as exc:
            raise ModelNotReadyError(f"MLX generation failed: {exc}") from exc

        text = output.text if hasattr(output, "text") else str(output)
        return BackendResponse(
            text=text,
            metadata={
                "runtime": "mlx_vlm",
                "model_source": self.config.local_model_path or self.config.model_id,
                "model_revision": self.config.revision,
            },
        )
