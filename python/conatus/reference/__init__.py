"""NumPy reference backend: the oracle of every fast path (C++/CUDA banded solves, NumPy/CuPy transforms
and fused kernels; docs/ARCHITECTURE.md)."""
from .evaluate import FieldData, ReferenceEvaluator, evaluate
from .transforms import grid  # collocation points of a basis (dealiased or not, explicit)

__all__ = [
    "FieldData", "ReferenceEvaluator", "evaluate", "grid",
]
