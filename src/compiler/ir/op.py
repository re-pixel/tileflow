from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .dtypes import DType, float32

if TYPE_CHECKING:
	from .tensor import Tensor


class IRValidationError(ValueError):
	pass


@dataclass(slots=True)
class Op:
	"""Base class for IR operations.

	Note: `output` may be `None` for ops constructed manually. When an op is
	created through `Graph` (e.g. `Graph.matmul(...)`), the graph is responsible
	for creating the output tensor and assigning `op.output`.
	"""

	name: str
	inputs: list[Tensor]
	attrs: dict[str, object] = field(default_factory=dict)
	output: Tensor | None = None

	@property
	def kind(self) -> str:
		return self.__class__.__name__

	def infer_output(self) -> tuple[tuple[int, ...], DType]:
		raise NotImplementedError


@dataclass(slots=True)
class MatMul(Op):
	"""Matrix multiplication: (M,K) @ (K,N) -> (M,N)."""

	def infer_output(self) -> tuple[tuple[int, ...], DType]:
		if len(self.inputs) != 2:
			raise IRValidationError("MatMul expects exactly 2 inputs")
		a, b = self.inputs
		if a.rank != 2 or b.rank != 2:
			raise IRValidationError("MatMul currently supports only rank-2 tensors")
		m, k1 = a.shape
		k2, n = b.shape
		if k1 != k2:
			raise IRValidationError(f"MatMul K mismatch: {k1} != {k2}")
		if a.dtype != b.dtype:
			raise IRValidationError("MatMul dtype mismatch")
		if a.dtype != float32:
			raise IRValidationError("Only float32 is supported for now")
		return (m, n), a.dtype


@dataclass(slots=True)
class Add(Op):
	"""Elementwise add: same-shape tensors only (no broadcasting yet)."""

	def infer_output(self) -> tuple[tuple[int, ...], DType]:
		if len(self.inputs) != 2:
			raise IRValidationError("Add expects exactly 2 inputs")
		a, b = self.inputs
		if a.shape != b.shape:
			raise IRValidationError("Add currently requires identical shapes (no broadcasting)")
		if a.dtype != b.dtype:
			raise IRValidationError("Add dtype mismatch")
		return a.shape, a.dtype


@dataclass(slots=True)
class ReLU(Op):
	"""Elementwise ReLU: output spec equals input spec."""

	def infer_output(self) -> tuple[tuple[int, ...], DType]:
		if len(self.inputs) != 1:
			raise IRValidationError("ReLU expects exactly 1 input")
		x = self.inputs[0]
		return x.shape, x.dtype


@dataclass(slots=True)
class FusedMatMulReLU(MatMul):
	"""Fused MatMul + ReLU: (M,K) @ (K,N) -> (M,N) followed by ReLU."""
	pass
