from __future__ import annotations

# Allow running via: `python -m examples.mlp_tiled` from repo root
# without requiring an editable install.
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from compiler.ir import Graph  # noqa: E402
from compiler.passes.tiling import TilingPass  # noqa: E402


def build_3layer_mlp(hidden: int = 128) -> Graph:
    """Build a tiny 3-layer MLP graph (MatMul + ReLU repeated).

    Shapes are chosen so a 128x128 MatMul appears (the Week 1 deliverable).
    """

    g = Graph(name="mlp_3layer")

    x = g.input("x", (hidden, hidden))
    w1 = g.param("w1", (hidden, hidden))
    w2 = g.param("w2", (hidden, hidden))
    w3 = g.param("w3", (hidden, hidden))

    h1 = g.matmul(x, w1, name="mm1", output_name="h1")
    a1 = g.relu(h1, name="relu1", output_name="a1")

    h2 = g.matmul(a1, w2, name="mm2", output_name="h2")
    a2 = g.relu(h2, name="relu2", output_name="a2")

    _y = g.matmul(a2, w3, name="mm3", output_name="y")
    return g

def show_graph(g: Graph) -> None:
    print(g.summary())
    print("\nTiled Graph:")

    tiler = TilingPass()
    for line in tiler.run(g):
        print(f"- {line}")

    print('\n')


def main() -> None:
    print("MLP with no padding (128x128 MatMuls): \n")
    g_no_padding = build_3layer_mlp(128)
    show_graph(g_no_padding)
    print("MLP with padding (130x130 MatMuls): \n")
    g_with_padding = build_3layer_mlp(130)
    show_graph(g_with_padding)


if __name__ == "__main__":
    main()
