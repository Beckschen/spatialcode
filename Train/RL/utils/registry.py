"""
Component registry with config-driven building.

Two build modes:
    # Mode 1: Config-driven (preferred)
    encoder = BACKBONES.build(SAM2Config(model_name="..."))

    # Mode 2: Legacy kwargs
    encoder = BACKBONES.build("sam2", model_name="...")

Custom registration:
    @BACKBONES.register("my_backbone")
    class MyBackbone(nn.Module):
        def __init__(self, cfg: MyBackboneConfig):
            ...
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class Registry:
    """A typed string -> class registry with config-driven build.

    Args:
        name: Human-readable name for error messages.
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    # ── Registration ──────────────────────────────────────────────────

    def register(self, name: Optional[str] = None) -> Callable:
        """Decorator to register a class.

        Usage:
            @BACKBONES.register("sam2")
            class SAM2Encoder(nn.Module):
                def __init__(self, cfg: SAM2Config): ...

            @BACKBONES.register()  # uses class.__name__
            class MyEncoder(nn.Module): ...
        """
        def decorator(cls_or_fn):
            key = _name or cls_or_fn.__name__
            if key in self._registry:
                raise KeyError(f"{self._name}: '{key}' is already registered.")
            self._registry[key] = cls_or_fn
            return cls_or_fn

        # Handle @registry.register (no parens) vs @registry.register("name")
        if callable(name):
            fn = name
            _name = None
            name = None
            return decorator(fn)

        _name = name
        return decorator

    # ── Building ──────────────────────────────────────────────────────

    def build(self, cfg_or_name, **kwargs) -> Any:
        """Build a component from config or name.

        Args:
            cfg_or_name: Either a BaseConfig with a `type` field,
                         or a string registry key.
            **kwargs:    Override fields (merged into config or passed directly).

        Returns:
            Instantiated component.

        Examples:
            # Config-driven
            BACKBONES.build(SAM2Config(model_name="facebook/sam2-hiera-base"))

            # String + kwargs (legacy)
            BACKBONES.build("sam2", model_name="facebook/sam2-hiera-base")
        """
        from spatialcode.config import BaseConfig

        if isinstance(cfg_or_name, BaseConfig):
            # Config-driven build
            cfg = cfg_or_name
            if kwargs:
                cfg = cfg.merge(kwargs)
            type_key = cfg.type
            cls = self._get(type_key)
            return cls(cfg)

        elif isinstance(cfg_or_name, str):
            # Legacy string build
            cls = self._get(cfg_or_name)
            return cls(**kwargs)

        elif isinstance(cfg_or_name, dict):
            # Dict build (from YAML): {"type": "sam2", "model_name": "..."}
            d = dict(cfg_or_name)
            type_key = d.pop("type")
            cls = self._get(type_key)
            # Try to find config class from CONFIG_REGISTRY
            from spatialcode.config import CONFIG_REGISTRY
            if type_key in CONFIG_REGISTRY:
                cfg_cls = CONFIG_REGISTRY[type_key]
                cfg = cfg_cls.from_dict({**d, "type": type_key})
                return cls(cfg)
            return cls(**d)

        else:
            raise TypeError(
                f"{self._name}.build() expects a BaseConfig, str, or dict, "
                f"got {type(cfg_or_name)}"
            )

    # ── Lookup ────────────────────────────────────────────────────────

    def _get(self, name: str) -> Any:
        if name not in self._registry:
            avail = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"{self._name}: '{name}' not found. Available: [{avail}]"
            )
        return self._registry[name]

    def get(self, name: str) -> Any:
        return self._get(name)

    def keys(self):
        return self._registry.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry('{self._name}', keys={list(self._registry.keys())})"


# ═════════════════════════════════════════════════════════════════════
# Global Registries
# ═════════════════════════════════════════════════════════════════════

BACKBONES = Registry("backbones")
FUSIONS = Registry("fusions")
HEADS = Registry("heads")
TRACKERS = Registry("trackers")
LLM_BACKENDS = Registry("llm_backends")
