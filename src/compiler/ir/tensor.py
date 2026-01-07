from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from .dtypes import DType

if TYPE_CHECKING:
	from .graph import Graph
	from .op import Op


Shape = tuple[int, ...]


@dataclass(slots=True)
class Tensor:
	"""A node value in the graph.

	Think of a Tensor as an SSA value: it has a single producer op (or None for
	inputs/params) and a list of ops that consume it.
	"""

	graph: Graph
	name: str
	shape: Shape
	dtype: DType
	producer: Op | None = None
	users: list[Op] = field(default_factory=list)

	def add_user(self, op: Op) -> None:
		self.users.append(op)

	@property
	def rank(self) -> int:
		return len(self.shape)

	@property
	def numel(self) -> int:
		n = 1
		for dim in self.shape:
			n *= dim
		return n

	def __repr__(self) -> str:  # pragma: no cover
		return f"Tensor(name={self.name!r}, shape={self.shape}, dtype={self.dtype})"


def as_shape(dims: Iterable[int]) -> Shape:
	return tuple(int(d) for d in dims)
