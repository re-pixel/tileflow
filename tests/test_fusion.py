from compiler.ir import FusedMatMulReLU, Graph, MatMul, ReLU
from compiler.passes.fusion import FusionPass


def test_fusion_simple() -> None:
    # A -> MatMul -> B -> ReLU -> C
    g = Graph(name="test_fusion")
    a = g.input("a", (32, 32))
    b = g.param("b", (32, 32))
    
    mm_out = g.matmul(a, b, name="mm")
    relu_out = g.relu(mm_out, name="relu")
    
    # Pre-check
    assert len(g.ops) == 2
    assert isinstance(g.ops[0], MatMul)
    assert isinstance(g.ops[1], ReLU)
    assert mm_out.users == [g.ops[1]]
    
    # Run Fusion
    FusionPass().run(g)
    
    # Verify
    assert len(g.ops) == 1
    op = g.ops[0]
    assert isinstance(op, FusedMatMulReLU)
    assert op.inputs == [a, b]
    assert op.output is relu_out
    assert relu_out.producer is op
    # Check that 'a' now points to fused op
    assert a.users == [op]


def test_fusion_multiple_users_no_fuse() -> None:
    # A -> MatMul -> B
    #           |-> ReLU -> C
    #           |-> (other use)
    g = Graph(name="test_no_fusion")
    a = g.input("a", (32, 32))
    b = g.param("b", (32, 32))
    
    mm_out = g.matmul(a, b, name="mm")
    relu_out = g.relu(mm_out, name="relu")
    # Add another user for mm_out
    _ = g.relu(mm_out, name="relu2")
    
    # Expect 3 ops: MatMul, ReLU, ReLU
    assert len(g.ops) == 3
    
    FusionPass().run(g)
    
    # Should NOT fuse because mm_out has 2 users
    assert len(g.ops) == 3
    assert isinstance(g.ops[0], MatMul)


def test_fusion_chain() -> None:
    # MM -> ReLU -> MM -> ReLU
    # Should become Fused -> Fused
    g = Graph(name="test_chain")
    a = g.input("a", (32, 32))
    b = g.param("b", (32, 32))
    
    h1 = g.matmul(a, b, name="mm1")
    a1 = g.relu(h1, name="relu1")
    
    h2 = g.matmul(a1, b, name="mm2")
    a2 = g.relu(h2, name="relu2")
    
    FusionPass().run(g)
    
    assert len(g.ops) == 2
    assert isinstance(g.ops[0], FusedMatMulReLU)
    assert isinstance(g.ops[1], FusedMatMulReLU)
    # Check connectivity
    # ops[0] output should be input to ops[1]
    op1 = g.ops[0]
    op2 = g.ops[1]
    assert op1.output in op2.inputs
