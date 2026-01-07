"""mini-compiler: Python frontend (IR + passes).

The Python side is intentionally small and explicit: we model a tiny graph IR and
run transformation/analysis passes over it.
"""

from .ir.dtypes import DType, float32
from .ir.graph import Graph
from .ir.tensor import Tensor
from .ir.op import IRValidationError

__all__ = ["DType", "float32", "Graph", "Tensor", "IRValidationError"]
