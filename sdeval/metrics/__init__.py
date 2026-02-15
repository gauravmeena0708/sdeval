from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from sdeval.config import EvaluatorSettings


@dataclass
class MetricContext:
    """Shared data passed to every metric."""

    real_df: pd.DataFrame
    synthetic_df: pd.DataFrame
    settings: "EvaluatorSettings"
    synthetic_path: str


class MetricCallable(Protocol):
    def __call__(self, ctx: MetricContext) -> Dict[str, Any]:
        ...


REGISTRY: Dict[str, MetricCallable] = {}


def register_metric(name: str) -> Callable[[MetricCallable], MetricCallable]:
    """Decorator used by metric modules to register themselves."""

    def decorator(fn: MetricCallable) -> MetricCallable:
        REGISTRY[name] = fn
        return fn

    return decorator


def get_metric(name: str) -> Optional[MetricCallable]:
    return REGISTRY.get(name)


# Import side effects register built-in metrics.
from . import statistical  # noqa: E402,F401
from . import constraints  # noqa: E402,F401
from . import coverage  # noqa: E402,F401
from . import ml_efficacy  # noqa: E402,F401
from . import plausibility  # noqa: E402,F401
from . import privacy  # noqa: E402,F401
from . import dp  # noqa: E402,F401

