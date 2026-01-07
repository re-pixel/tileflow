from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DType:
	"""Scalar dtype for IR tensors.

	We keep this intentionally tiny because the project constraint is fixed:
	32x32 tiles of float32.
	"""

	name: str
	itemsize: int

	def __str__(self) -> str:  # pragma: no cover
		return self.name


float32 = DType("float32", 4)
