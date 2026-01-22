"""
Unit tests for AVX2 kernel correctness via Python bindings.

These tests verify that the AVX2 kernels produce correct results
when called through the pybind11 interface.
"""

import numpy as np
import pytest

try:
    import mini_runtime as rt
except ImportError:
    pytest.skip("mini_runtime not built", allow_module_level=True)


TILE_DIM = 32
RTOL = 1e-4
ATOL = 1e-5


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def random_tiles(rng):
    """Generate random A, B, C tiles."""
    A = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32) * 0.1)
    B = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32) * 0.1)
    C = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
    return A, B, C


class TestKernelAvailability:
    """Test kernel availability queries."""
    
    def test_is_avx2_available_returns_bool(self):
        result = rt.is_avx2_available()
        assert isinstance(result, bool)
    
    def test_get_active_kernel_name_returns_string(self):
        name = rt.get_active_kernel_name()
        assert isinstance(name, str)
        assert name in ("AVX2+FMA", "Reference")
    
    def test_kernel_impl_enum_exists(self):
        assert hasattr(rt, "KernelImpl")
        assert hasattr(rt.KernelImpl, "Reference")
        assert hasattr(rt.KernelImpl, "AVX2")


class TestReferenceKernel:
    """Test reference kernel correctness."""
    
    def test_identity_matrix_left(self, rng):
        """A = I, B = random => C = B"""
        A = np.ascontiguousarray(np.eye(TILE_DIM, dtype=np.float32))
        B = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32))
        C = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
        
        rt.matmul_tile_bench(C, A, B, rt.KernelImpl.Reference)
        
        np.testing.assert_allclose(C, B, rtol=RTOL, atol=ATOL)
    
    def test_identity_matrix_right(self, rng):
        """A = random, B = I => C = A"""
        A = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32))
        B = np.ascontiguousarray(np.eye(TILE_DIM, dtype=np.float32))
        C = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
        
        rt.matmul_tile_bench(C, A, B, rt.KernelImpl.Reference)
        
        np.testing.assert_allclose(C, A, rtol=RTOL, atol=ATOL)
    
    def test_accumulation(self):
        """C starts non-zero, should accumulate."""
        A = np.ascontiguousarray(np.eye(TILE_DIM, dtype=np.float32))
        B = np.ascontiguousarray(np.eye(TILE_DIM, dtype=np.float32))
        C = np.ascontiguousarray(np.ones((TILE_DIM, TILE_DIM), dtype=np.float32))
        
        rt.matmul_tile_bench(C, A, B, rt.KernelImpl.Reference)
        
        # C should be 1 + I
        expected = np.ones((TILE_DIM, TILE_DIM), dtype=np.float32) + np.eye(TILE_DIM, dtype=np.float32)
        np.testing.assert_allclose(C, expected, rtol=RTOL, atol=ATOL)
    
    def test_matches_numpy(self, random_tiles):
        """Reference kernel should match numpy."""
        A, B, C = random_tiles
        
        rt.matmul_tile_bench(C, A, B, rt.KernelImpl.Reference)
        
        expected = A @ B
        np.testing.assert_allclose(C, expected, rtol=RTOL, atol=ATOL)


class TestAVX2Kernel:
    """Test AVX2 kernel correctness."""
    
    def test_matches_reference(self, random_tiles):
        """AVX2 kernel should match reference kernel."""
        A, B, C = random_tiles
        
        C_ref = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
        C_avx2 = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
        
        rt.matmul_tile_bench(C_ref, A, B, rt.KernelImpl.Reference)
        rt.matmul_tile_bench(C_avx2, A, B, rt.KernelImpl.AVX2)
        
        np.testing.assert_allclose(C_avx2, C_ref, rtol=RTOL, atol=ATOL)
    
    def test_matches_numpy(self, random_tiles):
        """AVX2 kernel should match numpy."""
        A, B, C = random_tiles
        
        rt.matmul_tile_bench(C, A, B, rt.KernelImpl.AVX2)
        
        expected = A @ B
        np.testing.assert_allclose(C, expected, rtol=RTOL, atol=ATOL)
    
    def test_identity_matrix(self, rng):
        """A = I => C = B"""
        A = np.ascontiguousarray(np.eye(TILE_DIM, dtype=np.float32))
        B = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32))
        C = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
        
        rt.matmul_tile_bench(C, A, B, rt.KernelImpl.AVX2)
        
        np.testing.assert_allclose(C, B, rtol=RTOL, atol=ATOL)
    
    def test_accumulation_matches_reference(self, rng):
        """Accumulation should match between implementations."""
        A = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32) * 0.1)
        B = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32) * 0.1)
        C_init = rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32) * 0.01
        
        C_ref = np.ascontiguousarray(C_init.copy())
        C_avx2 = np.ascontiguousarray(C_init.copy())
        
        rt.matmul_tile_bench(C_ref, A, B, rt.KernelImpl.Reference)
        rt.matmul_tile_bench(C_avx2, A, B, rt.KernelImpl.AVX2)
        
        np.testing.assert_allclose(C_avx2, C_ref, rtol=1e-4, atol=1e-5)
    
    def test_large_values(self, rng):
        """Numerical stability with large values."""
        A = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32) * 100)
        B = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32) * 100)
        C_ref = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
        C_avx2 = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
        
        rt.matmul_tile_bench(C_ref, A, B, rt.KernelImpl.Reference)
        rt.matmul_tile_bench(C_avx2, A, B, rt.KernelImpl.AVX2)
        
        # Use relative tolerance for large values
        np.testing.assert_allclose(C_avx2, C_ref, rtol=1e-4, atol=1e-2)
    
    def test_small_values(self, rng):
        """Numerical stability with small values."""
        A = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32) * 1e-4)
        B = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32) * 1e-4)
        C_ref = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
        C_avx2 = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
        
        rt.matmul_tile_bench(C_ref, A, B, rt.KernelImpl.Reference)
        rt.matmul_tile_bench(C_avx2, A, B, rt.KernelImpl.AVX2)
        
        np.testing.assert_allclose(C_avx2, C_ref, rtol=1e-4, atol=1e-10)


class TestAutoDispatch:
    """Test auto-dispatched kernels."""
    
    def test_auto_dispatch_matches_numpy(self, random_tiles):
        """Auto-dispatched kernel should match numpy."""
        A, B, C = random_tiles
        
        rt.matmul_tile_bench(C, A, B)  # No impl argument = auto dispatch
        
        expected = A @ B
        np.testing.assert_allclose(C, expected, rtol=RTOL, atol=ATOL)


class TestReLU:
    """Test ReLU kernel."""
    
    def test_relu_positive_values(self):
        """Positive values should remain unchanged."""
        C = np.ascontiguousarray(np.arange(1, TILE_DIM * TILE_DIM + 1, dtype=np.float32).reshape(TILE_DIM, TILE_DIM))
        expected = C.copy()
        
        rt.relu_tile_bench(C, rt.KernelImpl.Reference)
        
        np.testing.assert_array_equal(C, expected)
    
    def test_relu_negative_values(self):
        """Negative values should become zero."""
        C = np.ascontiguousarray(-np.arange(1, TILE_DIM * TILE_DIM + 1, dtype=np.float32).reshape(TILE_DIM, TILE_DIM))
        
        rt.relu_tile_bench(C, rt.KernelImpl.Reference)
        
        assert np.all(C == 0)
    
    def test_relu_mixed_values(self, rng):
        """Mixed values should be clamped correctly."""
        C = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32))
        expected = np.maximum(C, 0)
        
        rt.relu_tile_bench(C, rt.KernelImpl.Reference)
        
        np.testing.assert_array_equal(C, expected)
    
    def test_relu_avx2_matches_reference(self, rng):
        """AVX2 ReLU should match reference."""
        data = rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32)
        C_ref = np.ascontiguousarray(data.copy())
        C_avx2 = np.ascontiguousarray(data.copy())
        
        rt.relu_tile_bench(C_ref, rt.KernelImpl.Reference)
        rt.relu_tile_bench(C_avx2, rt.KernelImpl.AVX2)
        
        np.testing.assert_array_equal(C_avx2, C_ref)


class TestInputValidation:
    """Test input validation."""
    
    def test_wrong_dimensions(self):
        """Should reject non-32x32 arrays."""
        A = np.zeros((16, 16), dtype=np.float32)
        B = np.zeros((16, 16), dtype=np.float32)
        C = np.zeros((16, 16), dtype=np.float32)
        
        with pytest.raises(ValueError, match="32x32"):
            rt.matmul_tile_bench(C, A, B)
    
    def test_non_2d_array(self):
        """Should reject non-2D arrays."""
        A = np.zeros((32, 32, 1), dtype=np.float32)
        B = np.zeros((32, 32), dtype=np.float32)
        C = np.zeros((32, 32), dtype=np.float32)
        
        with pytest.raises(ValueError, match="2D"):
            rt.matmul_tile_bench(C, A, B)


@pytest.mark.skipif(not rt.is_avx2_available(), reason="AVX2 not available")
class TestPerformance:
    """Performance sanity tests (only run if AVX2 is available)."""
    
    def test_avx2_faster_than_reference(self, rng):
        """AVX2 should be meaningfully faster than reference."""
        import time
        
        A = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32))
        B = np.ascontiguousarray(rng.standard_normal((TILE_DIM, TILE_DIM)).astype(np.float32))
        C = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
        
        WARMUP = 100
        ITERS = 1000
        
        # Warmup
        for _ in range(WARMUP):
            rt.matmul_tile_bench(C, A, B, rt.KernelImpl.Reference)
            rt.matmul_tile_bench(C, A, B, rt.KernelImpl.AVX2)
        
        # Time reference
        start = time.perf_counter()
        for _ in range(ITERS):
            rt.matmul_tile_bench(C, A, B, rt.KernelImpl.Reference)
        ref_time = time.perf_counter() - start
        
        # Time AVX2
        start = time.perf_counter()
        for _ in range(ITERS):
            rt.matmul_tile_bench(C, A, B, rt.KernelImpl.AVX2)
        avx2_time = time.perf_counter() - start
        
        speedup = ref_time / avx2_time
        
        # AVX2 should be at least 2x faster (conservative for CI)
        assert speedup > 2.0, f"AVX2 speedup only {speedup:.2f}x (expected > 2x)"
