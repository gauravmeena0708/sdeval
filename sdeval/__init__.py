"""
Synthetic Data Evaluator package.

This package exposes a CLI entry point (``python -m sdeval.main``) plus a
programmatic API for running constraint-aware evaluations of synthetic
tabular datasets.
"""

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("sdeval")
except PackageNotFoundError:  # pragma: no cover - when running from source
    __version__ = "0.0.0"


__all__ = ["__version__"]
