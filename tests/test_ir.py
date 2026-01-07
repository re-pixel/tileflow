import pytest

from compiler.ir import Graph
from compiler.ir.op import IRValidationError


def test_matmul_shape_inference() -> None:
	g = Graph(name="shape")
	a = g.input("a", (128, 64))
	b = g.param("b", (64, 32))
	c = g.matmul(a, b)
	assert c.shape == (128, 32)
	assert len(g.ops) == 1


def test_add_and_relu_shapes() -> None:
	g = Graph(name="elemwise")
	x = g.input("x", (32, 32))
	y = g.input("y", (32, 32))
	z = g.add(x, y)
	r = g.relu(z)
	assert z.shape == (32, 32)
	assert r.shape == (32, 32)
	assert len(g.ops) == 2


def test_cannot_mix_tensors_from_different_graphs() -> None:
	g1 = Graph(name="g1")
	g2 = Graph(name="g2")

	a = g1.input("a", (32, 32))
	b = g2.param("b", (32, 32))

	with pytest.raises(IRValidationError):
		_ = g1.matmul(a, b)
