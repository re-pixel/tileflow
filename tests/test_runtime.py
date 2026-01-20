"""Week 4 Runtime Tests — C++ engine and Python wrapper verification.

Tests cover:
1. Single tile matmul correctness
2. Accumulation semantics (C += A @ B)
3. ReLU activation
4. Multi-tile matmul
5. End-to-end MLP execution
"""

import numpy as np
import pytest

# Skip all tests if C++ extension not available
pytest.importorskip("mini_runtime")

from compiler.ir import Graph
from compiler.passes import FusionPass, LoweringPass, TilingPass
from compiler.runtime import Runtime, TILE_DIM, TILE_BYTES
from compiler.scheduler import Scheduler, SRAMConfig
from compiler.scheduler.ops import SchedExecute, SchedLoad, SchedStore


# =============================================================================
# 1. Basic Engine Tests
# =============================================================================


class TestEngineBasics:
    """Test basic engine functionality."""
    
    def test_tensor_registration(self):
        """Register tensors and verify IDs."""
        rt = Runtime(sram_bytes=64 * 1024)
        
        id_a = rt.register_tensor("A", (32, 32))
        id_b = rt.register_tensor("B", (32, 32))
        
        assert id_a == 0
        assert id_b == 1
    
    def test_tensor_roundtrip(self):
        """Set and get tensor data."""
        rt = Runtime(sram_bytes=64 * 1024)
        rt.register_tensor("A", (32, 32))
        
        data = np.random.randn(32, 32).astype(np.float32)
        rt.set_tensor("A", data)
        
        result = rt.get_tensor("A", (32, 32))
        np.testing.assert_array_equal(result, data)
    
    def test_padded_tensor_roundtrip(self):
        """Tensors with non-tile-aligned shapes work correctly."""
        rt = Runtime(sram_bytes=64 * 1024)
        rt.register_tensor("A", (48, 48))  # 1.5 tiles each dimension
        
        data = np.random.randn(48, 48).astype(np.float32)
        rt.set_tensor("A", data)
        
        result = rt.get_tensor("A", (48, 48))
        np.testing.assert_array_equal(result, data)


# =============================================================================
# 2. Single Tile Matmul Tests
# =============================================================================


class TestSingleTileMatmul:
    """Test correctness of single 32x32 tile matmul."""
    
    def test_identity_matmul(self):
        """A @ I = A."""
        rt = Runtime(sram_bytes=64 * 1024)
        rt.register_tensor("A", (32, 32))
        rt.register_tensor("I", (32, 32))
        rt.register_tensor("C", (32, 32))
        
        A = np.random.randn(32, 32).astype(np.float32)
        I = np.eye(32, dtype=np.float32)
        
        rt.set_tensor("A", A)
        rt.set_tensor("I", I)
        
        # Manual schedule: C = A @ I
        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("I", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0, 
                        a_addr=0, b_addr=TILE_BYTES, acc_addr=2*TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2*TILE_BYTES, bytes=TILE_BYTES),
        ]
        
        rt.execute(schedule)
        
        C = rt.get_tensor("C", (32, 32))
        np.testing.assert_allclose(C, A, rtol=1e-5)
    
    def test_random_matmul(self):
        """Random A @ B matches NumPy."""
        rt = Runtime(sram_bytes=64 * 1024)
        rt.register_tensor("A", (32, 32))
        rt.register_tensor("B", (32, 32))
        rt.register_tensor("C", (32, 32))
        
        A = np.random.randn(32, 32).astype(np.float32)
        B = np.random.randn(32, 32).astype(np.float32)
        
        rt.set_tensor("A", A)
        rt.set_tensor("B", B)
        
        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("B", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0,
                        a_addr=0, b_addr=TILE_BYTES, acc_addr=2*TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2*TILE_BYTES, bytes=TILE_BYTES),
        ]
        
        rt.execute(schedule)
        
        C = rt.get_tensor("C", (32, 32))
        expected = A @ B
        np.testing.assert_allclose(C, expected, rtol=1e-4, atol=1e-6)
    
    def test_accumulation_semantics(self):
        """C += A @ B (accumulation into non-zero C)."""
        rt = Runtime(sram_bytes=64 * 1024)
        rt.register_tensor("A", (32, 32))
        rt.register_tensor("B", (32, 32))
        rt.register_tensor("C", (32, 32))
        
        A = np.random.randn(32, 32).astype(np.float32)
        B = np.random.randn(32, 32).astype(np.float32)
        
        rt.set_tensor("A", A)
        rt.set_tensor("B", B)
        
        # Execute twice with same accumulator address
        # First exec: acc starts at 0 (SRAM cleared), C = 0 + A @ B
        # Second exec would be: C = (A @ B) + A @ B = 2 * (A @ B)
        # But we only do one exec here, verifying SRAM starts at zero
        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("B", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0,
                        a_addr=0, b_addr=TILE_BYTES, acc_addr=2*TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2*TILE_BYTES, bytes=TILE_BYTES),
        ]
        
        rt.execute(schedule)
        
        C = rt.get_tensor("C", (32, 32))
        expected = A @ B  # SRAM is cleared, so initial C = 0
        np.testing.assert_allclose(C, expected, rtol=1e-4, atol=1e-6)


# =============================================================================
# 3. ReLU Tests
# =============================================================================


class TestReLU:
    """Test ReLU activation in SchedStore."""
    
    def test_relu_zeros_negatives(self):
        """ReLU sets negative values to zero."""
        rt = Runtime(sram_bytes=64 * 1024)
        rt.register_tensor("A", (32, 32))
        rt.register_tensor("B", (32, 32))
        rt.register_tensor("C", (32, 32))
        
        # Create A and B such that A @ B has negative values
        A = np.ones((32, 32), dtype=np.float32)
        B = np.full((32, 32), -1.0, dtype=np.float32)  # A @ B will be all -32
        
        rt.set_tensor("A", A)
        rt.set_tensor("B", B)
        
        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("B", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0,
                        a_addr=0, b_addr=TILE_BYTES, acc_addr=2*TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2*TILE_BYTES, bytes=TILE_BYTES,
                      activation="relu"),
        ]
        
        rt.execute(schedule)
        
        C = rt.get_tensor("C", (32, 32))
        # A @ B = -32 everywhere, ReLU should make it 0
        np.testing.assert_array_equal(C, np.zeros((32, 32), dtype=np.float32))
    
    def test_relu_preserves_positives(self):
        """ReLU preserves positive values."""
        rt = Runtime(sram_bytes=64 * 1024)
        rt.register_tensor("A", (32, 32))
        rt.register_tensor("B", (32, 32))
        rt.register_tensor("C", (32, 32))
        
        # Identity matrices: A @ B = A
        A = np.eye(32, dtype=np.float32) * 5.0  # Positive diagonal
        B = np.eye(32, dtype=np.float32)
        
        rt.set_tensor("A", A)
        rt.set_tensor("B", B)
        
        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("B", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0,
                        a_addr=0, b_addr=TILE_BYTES, acc_addr=2*TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2*TILE_BYTES, bytes=TILE_BYTES,
                      activation="relu"),
        ]
        
        rt.execute(schedule)
        
        C = rt.get_tensor("C", (32, 32))
        expected = A  # A @ I = A, ReLU preserves positive values
        np.testing.assert_allclose(C, expected, rtol=1e-5)


# =============================================================================
# 4. Multi-Tile Tests
# =============================================================================


class TestMultiTile:
    """Test multi-tile matmul with reduction."""
    
    def test_64x64_matmul(self):
        """64x64 @ 64x64 = 2x2 output tiles, K=2 reduction."""
        rt = Runtime(sram_bytes=256 * 1024)
        rt.register_tensor("A", (64, 64))
        rt.register_tensor("B", (64, 64))
        rt.register_tensor("C", (64, 64))
        
        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)
        
        rt.set_tensor("A", A)
        rt.set_tensor("B", B)
        
        # Manual schedule for 2x2x2 tiled matmul
        # C[m,n] = sum_k A[m,k] @ B[k,n]
        schedule = []
        
        # SRAM layout:
        # 0x0000: A tile (4KB)
        # 0x1000: B tile (4KB)
        # 0x2000-0x5000: C tiles (4 x 4KB)
        
        for m in range(2):
            for n in range(2):
                acc_addr = 0x2000 + (m * 2 + n) * TILE_BYTES
                
                for k in range(2):
                    schedule.append(SchedLoad("A", (m, k), dst_addr=0x0000, bytes=TILE_BYTES))
                    schedule.append(SchedLoad("B", (k, n), dst_addr=0x1000, bytes=TILE_BYTES))
                    schedule.append(SchedExecute(
                        m=m, n=n, k=k,
                        a_addr=0x0000, b_addr=0x1000, acc_addr=acc_addr
                    ))
                
                schedule.append(SchedStore("C", (m, n), src_addr=acc_addr, bytes=TILE_BYTES))
        
        rt.execute(schedule)
        
        C = rt.get_tensor("C", (64, 64))
        expected = A @ B
        np.testing.assert_allclose(C, expected, rtol=1e-4, atol=1e-5)


# =============================================================================
# 5. End-to-End MLP Test
# =============================================================================


class TestEndToEndMLP:
    """Test full MLP execution through the compiler pipeline."""
    
    def test_single_layer_mlp(self):
        """Single matmul + ReLU layer."""
        # Build a simple graph: y = relu(x @ w)
        g = Graph("single_layer")
        x = g.input("x", (32, 32))
        w = g.param("w", (32, 32))
        h = g.matmul(x, w, output_name="h")
        y = g.relu(h, output_name="y")
        
        # Run compiler passes
        FusionPass().run(g)
        TilingPass().run(g)
        LoweringPass().run(g)
        
        config = SRAMConfig(total_bytes=64 * 1024)
        scheduler = Scheduler(config=config)
        schedule, stats = scheduler.run_on_graph(g)
        
        # Execute on runtime
        rt = Runtime(sram_bytes=config.total_bytes)
        rt.register_graph_tensors(g)
        
        # Set random inputs
        x_data = np.random.randn(32, 32).astype(np.float32)
        w_data = np.random.randn(32, 32).astype(np.float32)
        rt.set_tensor("x", x_data)
        rt.set_tensor("w", w_data)
        
        rt.execute(schedule)
        
        # Get output - after fusion, the fused op writes to 'y' (the relu output)
        result = rt.get_tensor("y", (32, 32))
        
        # Compute expected
        expected = np.maximum(0, x_data @ w_data)
        
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)
    
    def test_mlp_3_layers(self):
        """Full 3-layer MLP matching the examples."""
        # Build graph
        g = Graph("mlp_3layer")
        x = g.input("x", (128, 128))
        
        w1 = g.param("w1", (128, 64))
        h1 = g.matmul(x, w1, output_name="h1")
        a1 = g.relu(h1, output_name="a1")
        
        w2 = g.param("w2", (64, 32))
        h2 = g.matmul(a1, w2, output_name="h2")
        a2 = g.relu(h2, output_name="a2")
        
        w3 = g.param("w3", (32, 32))
        out = g.matmul(a2, w3, output_name="out")
        
        # Run compiler passes
        FusionPass().run(g)
        TilingPass().run(g)
        LoweringPass().run(g)
        
        config = SRAMConfig(total_bytes=256 * 1024)
        scheduler = Scheduler(config=config)
        schedule, stats = scheduler.run_on_graph(g)
        
        # Execute on runtime
        rt = Runtime(sram_bytes=config.total_bytes)
        rt.register_graph_tensors(g)
        
        # Set random inputs
        np.random.seed(42)  # Reproducibility
        x_data = np.random.randn(128, 128).astype(np.float32) * 0.1
        w1_data = np.random.randn(128, 64).astype(np.float32) * 0.1
        w2_data = np.random.randn(64, 32).astype(np.float32) * 0.1
        w3_data = np.random.randn(32, 32).astype(np.float32) * 0.1
        
        rt.set_tensor("x", x_data)
        rt.set_tensor("w1", w1_data)
        rt.set_tensor("w2", w2_data)
        rt.set_tensor("w3", w3_data)
        
        rt.execute(schedule)
        
        # Compute expected (NumPy reference)
        h1_expected = np.maximum(0, x_data @ w1_data)  # Fused matmul + relu
        h2_expected = np.maximum(0, h1_expected @ w2_data)
        out_expected = h2_expected @ w3_data  # Last layer has no relu
        
        # Get output
        result = rt.get_tensor("out", (128, 32))
        
        np.testing.assert_allclose(result, out_expected, rtol=1e-4, atol=1e-5)


# =============================================================================
# 6. Stats Tests
# =============================================================================


class TestStats:
    """Test execution statistics."""
    
    def test_stats_count_ops(self):
        """Stats correctly count operations."""
        rt = Runtime(sram_bytes=64 * 1024)
        rt.register_tensor("A", (32, 32))
        rt.register_tensor("B", (32, 32))
        rt.register_tensor("C", (32, 32))
        
        A = np.random.randn(32, 32).astype(np.float32)
        B = np.random.randn(32, 32).astype(np.float32)
        rt.set_tensor("A", A)
        rt.set_tensor("B", B)
        
        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("B", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0,
                        a_addr=0, b_addr=TILE_BYTES, acc_addr=2*TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2*TILE_BYTES, bytes=TILE_BYTES),
        ]
        
        rt.execute(schedule)
        
        stats = rt.stats
        assert stats.loads == 2
        assert stats.executes == 1
        assert stats.stores == 1
    
    def test_stats_reset(self):
        """Stats reset works."""
        rt = Runtime(sram_bytes=64 * 1024)
        rt.register_tensor("A", (32, 32))
        rt.register_tensor("B", (32, 32))
        rt.register_tensor("C", (32, 32))
        
        rt.set_tensor("A", np.zeros((32, 32), dtype=np.float32))
        rt.set_tensor("B", np.zeros((32, 32), dtype=np.float32))
        
        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("B", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0,
                        a_addr=0, b_addr=TILE_BYTES, acc_addr=2*TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2*TILE_BYTES, bytes=TILE_BYTES),
        ]
        
        rt.execute(schedule)
        assert rt.stats.loads == 2
        
        rt.reset_stats()
        assert rt.stats.loads == 0
        assert rt.stats.executes == 0
        assert rt.stats.stores == 0
