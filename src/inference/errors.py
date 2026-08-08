"""Model-layer errors suitable for stable upstream error mapping."""


class ModelError(RuntimeError):
    """Base class for model-side failures."""


class ModelNotReadyError(ModelError):
    """The configured runtime, weights, or adapter are not locally ready."""


class ModelOutputError(ModelError):
    """The model output cannot be parsed or violates the model contract."""


class UnsupportedBackendError(ModelError):
    """The requested backend is not registered in this milestone."""
