from compiler.ir import Graph
from compiler.passes.tiling import HardwareConstraintValidator


def test_tiling_no_padding_128() -> None:
	g = Graph(name="tiling")
	a = g.input("a", (128, 128))
	b = g.param("b", (128, 128))
	_ = g.matmul(a, b, name="mm")

	HardwareConstraintValidator().run(g)

	mm = g.ops[0]
	info = mm.attrs["matmul"]
	assert info["pad"] == {"m": 0, "n": 0, "k": 0}
	assert info["tiles"] == {"m": 4, "n": 4, "k": 4}
	assert info["output_tile_ops"] == 16


def test_tiling_padding_is_computed() -> None:
	g = Graph(name="tiling_pad")
	a = g.input("a", (33, 65))
	b = g.param("b", (65, 35))
	_ = g.matmul(a, b, name="mm")

	HardwareConstraintValidator().run(g)

	mm = g.ops[0]
	info = mm.attrs["matmul"]
	assert info["m_padded"] == 64
	assert info["n_padded"] == 64
	assert info["k_padded"] == 96
	assert info["pad"] == {"m": 31, "n": 29, "k": 31}
	assert info["tiles"] == {"m": 2, "n": 2, "k": 3}
	assert info["output_tile_ops"] == 4
	assert info["mac_tile_ops"] == 12
