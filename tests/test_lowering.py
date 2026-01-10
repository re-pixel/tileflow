from compiler.ir import Graph
from compiler.passes import (
    FusionPass,
    HardwareConstraintValidator,
    LoweringPass,
    UOpExecute,
    UOpLoad,
    UOpStore,
)


def test_lower_simple_matmul() -> None:
    # 64x64 matmul (2x2 tiles)
    g = Graph(name="test_lower")
    a = g.input("a", (64, 64))
    b = g.param("b", (64, 64))
    g.matmul(a, b, name="mm")
    
    # Must tile first
    HardwareConstraintValidator().run(g)
    
    uops = LoweringPass().run(g)
    
    # 2x2 output tiles. Each needs full K reduction.
    # Grid: M=2, N=2, K=2. Total steps = 8.
    # Each step: Load A, Load B, Exec. (3 ops)
    # Plus Store per (m,n). Total 4 stores.
    # Total ops = 8*3 + 4 = 28?
    
    # Let's count types
    loads = [op for op in uops if isinstance(op, UOpLoad)]
    execs = [op for op in uops if isinstance(op, UOpExecute)]
    stores = [op for op in uops if isinstance(op, UOpStore)]
    
    # M=2, N=2, K=2
    # Inner loop runs 2*2*2 = 8 times.
    # 2 loads per inner it -> 16 loads.
    assert len(loads) == 16
    # 1 exec per inner it -> 8 execs.
    assert len(execs) == 8
    # Stores happen in N loop (2*2 = 4)
    assert len(stores) == 4
    
    # Check activation
    assert stores[0].activation is None


def test_lower_fused_matmul_relu() -> None:
    g = Graph(name="test_lower_fused")
    a = g.input("a", (32, 32))
    b = g.param("b", (32, 32))
    mm = g.matmul(a, b)
    g.relu(mm)
    
    FusionPass().run(g)
    HardwareConstraintValidator().run(g)
    uops = LoweringPass().run(g)
    
    stores = [op for op in uops if isinstance(op, UOpStore)]
    # 1 tile, 1 store
    assert len(stores) == 1
    assert stores[0].activation == "relu"
    # Output tensor should preserve the name from the original ReLU output
    assert stores[0].tensor == g.ops[0].output.name
    # Wait, in the test: g.relu creates an op. The tensor name comes from op.output.name.
    # In fusion, fused_op.output = relu.output.
    # So tensor name should be whatever relu output name is.
