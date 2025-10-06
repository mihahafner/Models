# All_Models/__init__.py
from importlib import import_module

_REGISTRY: dict[str, type] = {}

def register(name: str):
    """Decorator to register a model class under a short name."""
    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return _wrap

def get_model(name: str):
    """Return the registered class; lazy-import subpackage if needed."""
    if name not in _REGISTRY:
        # try to load All_Models/<name>/model.py lazily
        import_module(f"All_Models.{name}.model")
    if name not in _REGISTRY:
        raise KeyError(f"Model '{name}' not found. Is All_Models/{name}/model.py registered?")
    return _REGISTRY[name]

__all__ = ["register", "get_model"]
