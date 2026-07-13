from __future__ import annotations

from collections.abc import Callable

from statebudgetmem.core.experiment import MethodBuildContext
from statebudgetmem.core.method import MemoryMethod

MethodFactory = Callable[[MethodBuildContext], MemoryMethod]


class MethodRegistry:
    """Small explicit registry used by the unified runner."""

    def __init__(self) -> None:
        self._factories: dict[str, MethodFactory] = {}

    def register(self, name: str, factory: MethodFactory) -> None:
        normalized = name.strip()
        if not normalized:
            raise ValueError("method name must not be empty")
        if normalized in self._factories:
            raise ValueError(f"method already registered: {normalized}")
        self._factories[normalized] = factory

    def create(self, name: str, context: MethodBuildContext) -> MemoryMethod:
        try:
            return self._factories[name](context)
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise ValueError(
                f"unknown method: {name}; available methods: {available}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_method_registry() -> MethodRegistry:
    from statebudgetmem.baselines.tfidf.adapter import TfidfMemoryMethod

    registry = MethodRegistry()
    registry.register("tfidf_topk", lambda _context: TfidfMemoryMethod())
    return registry
