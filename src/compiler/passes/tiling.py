from __future__ import annotations

from dataclasses import dataclass

from compiler.ir.graph import Graph
from compiler.ir.op import MatMul, Op


def _ceil_div(a: int, b: int) -> int:
	return (a + b - 1) // b


def _pad_to_multiple(x: int, multiple: int) -> tuple[int, int]:
	"""Return (padded_x, pad_amount)."""

	if multiple <= 0:
		raise ValueError("multiple must be > 0")
	pad = (-x) % multiple
	return x + pad, pad


@dataclass(slots=True)
class HardwareConstraintValidator:
	"""Annotate ops with hardware-aware tiling + padding metadata.

	In Week 1, this pass does not rewrite the graph. It only computes what would
	be required to run on a fixed 32x32 tile accelerator.

	Why metadata instead of rewriting?
	- It keeps the IR simple while you're still learning the whole pipeline.
	- Week 2 lowering can consume these attrs to emit 32x32 micro-ops.
	"""

	tile_m: int = 32
	tile_n: int = 32
	tile_k: int = 32

	def run(self, graph: Graph) -> None:
		for op in graph.ops:
			if isinstance(op, MatMul):
				self._annotate_matmul(op)

		# Mark the graph as having valid tiling metadata.
		graph.attrs["tiling"] = {
			"validated": True,
			"tile": {"m": self.tile_m, "n": self.tile_n, "k": self.tile_k},
		}

	def _annotate_matmul(self, op: MatMul) -> None:
		# This pass only needs input shapes/dtypes; it should not require
		# `op.output` to be present (e.g., if an op was constructed manually).
		a, b = op.inputs
		m, k = a.shape
		_, n = b.shape

		m_pad, dm = _pad_to_multiple(m, self.tile_m)
		n_pad, dn = _pad_to_multiple(n, self.tile_n)
		k_pad, dk = _pad_to_multiple(k, self.tile_k)

		tm = _ceil_div(m_pad, self.tile_m)
		tn = _ceil_div(n_pad, self.tile_n)
		tk = _ceil_div(k_pad, self.tile_k)

		# Minimal, stable contract for later phases.
		op.attrs["tile"] = {"m": self.tile_m, "n": self.tile_n, "k": self.tile_k}
		op.attrs["matmul"] = {
			"m": m,
			"n": n,
			"k": k,
			"m_padded": m_pad,
			"n_padded": n_pad,
			"k_padded": k_pad,
			"pad": {"m": dm, "n": dn, "k": dk},
			"tiles": {"m": tm, "n": tn, "k": tk},
			# In the deliverable example we count output tiles.
			"output_tile_ops": tm * tn,
			# This is the full compute tile count if you unroll the K loop.
			"mac_tile_ops": tm * tn * tk,
		}

	@staticmethod
	def describe_op(op: Op) -> str:
		if not isinstance(op, MatMul):
			return f"{op.kind}({op.name})"
		info = op.attrs.get("matmul")
		tile = op.attrs.get("tile")
		if not isinstance(info, dict) or not isinstance(tile, dict):
			return f"MatMul({op.name})"
		m, n = info["m"], info["n"]
		tm, tn, tk = info["tiles"]["m"], info["tiles"]["n"], info["tiles"]["k"]
		out_ops = info["output_tile_ops"]
		dm, dn, dk = info["pad"]["m"], info["pad"]["n"], info["pad"]["k"]
		tile_m, tile_n = tile["m"], tile["n"]
		pad_part = ""
		if (dm, dn, dk) != (0, 0, 0):
			pad_part = f" (padding m+{dm}, n+{dn}, k+{dk})"
		return (
			f"This {m}x{n} MatMul will be executed as {out_ops} {tile_m}x{tile_n} tile ops"
			f" (tiles m={tm}, n={tn}, k_steps={tk}){pad_part}."
		)


@dataclass(slots=True)
class TilingPass:
	"""Convenience wrapper used by examples: validate and return descriptions."""

	validator: HardwareConstraintValidator = HardwareConstraintValidator()

	def run(self, graph: Graph) -> list[str]:
		self.validator.run(graph)
		return [self.validator.describe_op(op) for op in graph.ops if isinstance(op, MatMul)]
