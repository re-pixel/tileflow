from __future__ import annotations

from dataclasses import dataclass, field

from .dtypes import DType, float32
from .op import Add, MatMul, Op, ReLU
from .tensor import Tensor, as_shape


@dataclass
class Graph:
	"""A simple, explicit graph IR.

	Design choices (on purpose):
	- Ops are appended in creation order (we assume you build a DAG forward).
	- Tensors know their producer and users (needed for fusion and scheduling).
	- Graph is purely structural; passes attach metadata via op.attrs.
	"""

	name: str = "graph"
	ops: list[Op] = field(default_factory=list)
	tensors: list[Tensor] = field(default_factory=list)
	_name_counters: dict[str, int] = field(default_factory=dict)

	def _fresh_name(self, prefix: str) -> str:
		n = self._name_counters.get(prefix, 0) + 1
		self._name_counters[prefix] = n
		return f"{prefix}{n}"

	def _new_tensor(self, *, name: str | None, shape: tuple[int, ...], dtype: DType) -> Tensor:
		t = Tensor(graph=self, name=name or self._fresh_name("t"), shape=shape, dtype=dtype)
		self.tensors.append(t)
		return t

	def input(self, name: str, shape: tuple[int, ...], dtype: DType = float32) -> Tensor:
		return self._new_tensor(name=name, shape=as_shape(shape), dtype=dtype)

	def param(self, name: str, shape: tuple[int, ...], dtype: DType = float32) -> Tensor:
		return self._new_tensor(name=name, shape=as_shape(shape), dtype=dtype)

	def _add_op(self, op: Op, *, output_name: str | None = None) -> Tensor:
		for t in op.inputs:
			t.add_user(op)

		out_shape, out_dtype = op.infer_output()
		out = self._new_tensor(name=output_name, shape=out_shape, dtype=out_dtype)
		out.producer = op
		op.output = out

		self.ops.append(op)
		return out

	def matmul(self, a: Tensor, b: Tensor, *, name: str | None = None, output_name: str | None = None) -> Tensor:
		op = MatMul(name=name or self._fresh_name("matmul"), inputs=[a, b])
		return self._add_op(op, output_name=output_name)

	def add(self, a: Tensor, b: Tensor, *, name: str | None = None, output_name: str | None = None) -> Tensor:
		op = Add(name=name or self._fresh_name("add"), inputs=[a, b])
		return self._add_op(op, output_name=output_name)

	def relu(self, x: Tensor, *, name: str | None = None, output_name: str | None = None) -> Tensor:
		op = ReLU(name=name or self._fresh_name("relu"), inputs=[x])
		return self._add_op(op, output_name=output_name)

	def summary(self) -> str:
		lines: list[str] = [f"Graph(name={self.name!r}, ops={len(self.ops)}, tensors={len(self.tensors)})"]
		for op in self.ops:
			assert op.output is not None
			ins = ", ".join(f"{t.name}:{t.shape}" for t in op.inputs)
			out = f"{op.output.name}:{op.output.shape}"
			lines.append(f"- {op.name}: {op.kind}({ins}) -> {out}")
		return "\n".join(lines)
