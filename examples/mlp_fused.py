from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from compiler.ir import Graph, FusedMatMulReLU
from compiler.passes import FusionPass, HardwareConstraintValidator, LoweringPass

def build_3layer_mlp(hidden: int = 128) -> Graph:
    g = Graph(name="mlp_3layer")
    x = g.input("x", (hidden, hidden))
    w1 = g.param("w1", (hidden, hidden))
    w2 = g.param("w2", (hidden, hidden))
    w3 = g.param("w3", (hidden, hidden))

    h1 = g.matmul(x, w1, name="mm1")
    a1 = g.relu(h1, name="relu1")

    h2 = g.matmul(a1, w2, name="mm2")
    a2 = g.relu(h2, name="relu2")

    _ = g.matmul(a2, w3, name="mm3")
    return g

def main() -> None:
    print("Building MLP...")
    g = build_3layer_mlp(128)
    print(f"Original: {len(g.ops)} ops")
    print(g.summary())
    
    print("\nRunning Fusion...")
    FusionPass().run(g)
    print(f"Fused: {len(g.ops)} ops")
    
    for op in g.ops:
        if isinstance(op, FusedMatMulReLU):
            print(f"- Fused {op.name}")
        else:
            print(f"- {op.name} ({op.kind})")
            
    print("\nRunning Tiling Validator...")
    HardwareConstraintValidator().run(g)
    
    print("\nRunning Lowering...")
    uops = LoweringPass().run(g)
    print(f"Generated {len(uops)} micro-ops.")
    
    # Print sample
    print("First 20 uOps:")
    for op in uops[:20]:
        print(f"  {op}")

if __name__ == "__main__":
    main()
