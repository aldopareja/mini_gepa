"""Minimal local optimization package for gepa-opt-v2.

Exports:
- optimize (async): main entry point to run the local optimizer
- EvaluationBatch: adapter evaluation output type
"""

# Re-exports for convenience
from .core import optimize  # type: ignore
from .types import EvaluationBatch  # type: ignore

__all__ = ["optimize", "EvaluationBatch"]
