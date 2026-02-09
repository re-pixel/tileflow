"""Week 6 Tests — Threaded runtime correctness and stress testing.

Tests cover:
1. Threaded engine matches sequential engine (bit-identical output)
2. Single tile matmul in threaded mode
3. Multi-tile matmul in threaded mode
4. Full MLP end-to-end in threaded mode
5. Stress tests (repeated execution, no hangs or crashes)
"""

import time

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
# Helpers
# =============================================================================


def build_mlp_graph():
    """Build the standard 3-layer MLP graph."""
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

    return g


def compile_graph(g):
    """Run all compiler passes and return the schedule."""
    FusionPass().run(g)
    TilingPass().run(g)
    LoweringPass().run(g)

    config = SRAMConfig(total_bytes=256 * 1024)
    scheduler = Scheduler(config=config)
    schedule, stats = scheduler.run_on_graph(g)
    return schedule, stats, config


def set_mlp_data(rt, seed=42):
    """Set random input data for the MLP with a fixed seed."""
    np.random.seed(seed)
    x_data = np.random.randn(128, 128).astype(np.float32) * 0.1
    w1_data = np.random.randn(128, 64).astype(np.float32) * 0.1
    w2_data = np.random.randn(64, 32).astype(np.float32) * 0.1
    w3_data = np.random.randn(32, 32).astype(np.float32) * 0.1

    rt.set_tensor("x", x_data)
    rt.set_tensor("w1", w1_data)
    rt.set_tensor("w2", w2_data)
    rt.set_tensor("w3", w3_data)

    return x_data, w1_data, w2_data, w3_data


def numpy_reference(x, w1, w2, w3):
    """Compute MLP output using NumPy."""
    h1 = np.maximum(0, x @ w1)
    h2 = np.maximum(0, h1 @ w2)
    return h2 @ w3


# =============================================================================
# 1. Threaded Matches Sequential
# =============================================================================


class TestThreadedMatchesSequential:
    """Threaded engine produces identical output to sequential engine."""

    def test_single_tile_matches(self):
        """Single 32x32 tile matmul: threaded == sequential."""
        A = np.random.randn(32, 32).astype(np.float32)
        B = np.random.randn(32, 32).astype(np.float32)

        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("B", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0,
                         a_addr=0, b_addr=TILE_BYTES, acc_addr=2 * TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2 * TILE_BYTES, bytes=TILE_BYTES),
        ]

        results = {}
        for threaded in [False, True]:
            rt = Runtime(sram_bytes=64 * 1024, threaded=threaded)
            rt.register_tensor("A", (32, 32))
            rt.register_tensor("B", (32, 32))
            rt.register_tensor("C", (32, 32))
            rt.set_tensor("A", A)
            rt.set_tensor("B", B)
            rt.execute(schedule)
            results[threaded] = rt.get_tensor("C", (32, 32))

        np.testing.assert_array_equal(results[False], results[True])

    def test_mlp_matches(self):
        """Full MLP: threaded == sequential."""
        g = build_mlp_graph()
        schedule, stats, config = compile_graph(g)

        results = {}
        for threaded in [False, True]:
            rt = Runtime(sram_bytes=config.total_bytes, threaded=threaded)
            rt.register_graph_tensors(g)
            set_mlp_data(rt, seed=42)
            rt.execute(schedule)
            results[threaded] = rt.get_tensor("out", (128, 32))

        np.testing.assert_allclose(results[False], results[True],
                                   rtol=1e-5, atol=1e-6)


# =============================================================================
# 2. Threaded Single Tile
# =============================================================================


class TestThreadedSingleTile:
    """Test basic threaded operations on single tiles."""

    def test_identity_matmul(self):
        """A @ I = A in threaded mode."""
        rt = Runtime(sram_bytes=64 * 1024, threaded=True)
        rt.register_tensor("A", (32, 32))
        rt.register_tensor("I", (32, 32))
        rt.register_tensor("C", (32, 32))

        A = np.random.randn(32, 32).astype(np.float32)
        I = np.eye(32, dtype=np.float32)

        rt.set_tensor("A", A)
        rt.set_tensor("I", I)

        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("I", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0,
                         a_addr=0, b_addr=TILE_BYTES, acc_addr=2 * TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2 * TILE_BYTES, bytes=TILE_BYTES),
        ]

        rt.execute(schedule)

        C = rt.get_tensor("C", (32, 32))
        np.testing.assert_allclose(C, A, rtol=1e-5)

    def test_relu_in_threaded_mode(self):
        """ReLU works correctly in threaded store path."""
        rt = Runtime(sram_bytes=64 * 1024, threaded=True)
        rt.register_tensor("A", (32, 32))
        rt.register_tensor("B", (32, 32))
        rt.register_tensor("C", (32, 32))

        A = np.ones((32, 32), dtype=np.float32)
        B = np.full((32, 32), -1.0, dtype=np.float32)

        rt.set_tensor("A", A)
        rt.set_tensor("B", B)

        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("B", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0,
                         a_addr=0, b_addr=TILE_BYTES, acc_addr=2 * TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2 * TILE_BYTES, bytes=TILE_BYTES,
                       activation="relu"),
        ]

        rt.execute(schedule)

        C = rt.get_tensor("C", (32, 32))
        np.testing.assert_array_equal(C, np.zeros((32, 32), dtype=np.float32))


# =============================================================================
# 3. Threaded Multi-Tile
# =============================================================================


class TestThreadedMultiTile:
    """Test multi-tile operations in threaded mode."""

    def test_64x64_matmul(self):
        """64x64 matmul in threaded mode matches NumPy."""
        rt = Runtime(sram_bytes=256 * 1024, threaded=True)
        rt.register_tensor("A", (64, 64))
        rt.register_tensor("B", (64, 64))
        rt.register_tensor("C", (64, 64))

        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)

        rt.set_tensor("A", A)
        rt.set_tensor("B", B)

        schedule = []
        for m in range(2):
            for n in range(2):
                acc_addr = 0x2000 + (m * 2 + n) * TILE_BYTES
                for k in range(2):
                    schedule.append(SchedLoad("A", (m, k), dst_addr=0x0000,
                                              bytes=TILE_BYTES))
                    schedule.append(SchedLoad("B", (k, n), dst_addr=0x1000,
                                              bytes=TILE_BYTES))
                    schedule.append(SchedExecute(
                        m=m, n=n, k=k,
                        a_addr=0x0000, b_addr=0x1000, acc_addr=acc_addr))
                schedule.append(SchedStore("C", (m, n), src_addr=acc_addr,
                                           bytes=TILE_BYTES))

        rt.execute(schedule)

        C = rt.get_tensor("C", (64, 64))
        expected = A @ B
        np.testing.assert_allclose(C, expected, rtol=1e-4, atol=1e-5)


# =============================================================================
# 4. End-to-End MLP (Threaded)
# =============================================================================


class TestThreadedMLP:
    """Full MLP through compiler pipeline in threaded mode."""

    def test_mlp_3_layers(self):
        """3-layer MLP produces correct output in threaded mode."""
        g = build_mlp_graph()
        schedule, stats, config = compile_graph(g)

        rt = Runtime(sram_bytes=config.total_bytes, threaded=True)
        rt.register_graph_tensors(g)
        x_data, w1_data, w2_data, w3_data = set_mlp_data(rt)

        rt.execute(schedule)

        result = rt.get_tensor("out", (128, 32))
        expected = numpy_reference(x_data, w1_data, w2_data, w3_data)

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_stats_match(self):
        """Stats are correctly counted in threaded mode."""
        g = build_mlp_graph()
        schedule, _, config = compile_graph(g)

        results = {}
        for threaded in [False, True]:
            rt = Runtime(sram_bytes=config.total_bytes, threaded=threaded)
            rt.register_graph_tensors(g)
            set_mlp_data(rt)
            rt.execute(schedule)
            s = rt.stats
            results[threaded] = (s.loads, s.executes, s.stores)

        assert results[False] == results[True], (
            f"Stats mismatch: sequential={results[False]}, threaded={results[True]}"
        )


# =============================================================================
# 5. Stress Tests
# =============================================================================


class TestThreadedStress:
    """Stress tests — no crashes, no hangs, no data corruption."""

    def test_repeated_execution(self):
        """Run MLP 100 times in threaded mode with no crashes."""
        g = build_mlp_graph()
        schedule, _, config = compile_graph(g)

        rt = Runtime(sram_bytes=config.total_bytes, threaded=True)
        rt.register_graph_tensors(g)
        x_data, w1_data, w2_data, w3_data = set_mlp_data(rt)
        expected = numpy_reference(x_data, w1_data, w2_data, w3_data)

        for i in range(100):
            rt.reset_stats()
            rt.execute(schedule)

        # Verify final execution is still correct
        result = rt.get_tensor("out", (128, 32))
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_no_hang_single_tile(self):
        """Single tile execution completes within reasonable time."""
        rt = Runtime(sram_bytes=64 * 1024, threaded=True)
        rt.register_tensor("A", (32, 32))
        rt.register_tensor("B", (32, 32))
        rt.register_tensor("C", (32, 32))

        rt.set_tensor("A", np.ones((32, 32), dtype=np.float32))
        rt.set_tensor("B", np.ones((32, 32), dtype=np.float32))

        schedule = [
            SchedLoad("A", (0, 0), dst_addr=0, bytes=TILE_BYTES),
            SchedLoad("B", (0, 0), dst_addr=TILE_BYTES, bytes=TILE_BYTES),
            SchedExecute(m=0, n=0, k=0,
                         a_addr=0, b_addr=TILE_BYTES, acc_addr=2 * TILE_BYTES),
            SchedStore("C", (0, 0), src_addr=2 * TILE_BYTES, bytes=TILE_BYTES),
        ]

        start = time.monotonic()
        for _ in range(1000):
            rt.execute(schedule)
        elapsed = time.monotonic() - start

        # 1000 single-tile executions should complete well under 10 seconds
        assert elapsed < 10.0, f"Took {elapsed:.2f}s — possible hang or perf issue"


# =============================================================================
# 6. Benchmark (informational, not a strict test)
# =============================================================================


class TestThreadedBenchmark:
    """Performance comparison (informational)."""

    def test_threaded_speedup_or_parity(self):
        """Threaded mode should not be dramatically slower than sequential."""
        g = build_mlp_graph()
        schedule, _, config = compile_graph(g)

        n_iters = 50

        timings = {}
        for threaded in [False, True]:
            rt = Runtime(sram_bytes=config.total_bytes, threaded=threaded)
            rt.register_graph_tensors(g)
            set_mlp_data(rt)

            # Warm up
            rt.execute(schedule)

            start = time.monotonic()
            for _ in range(n_iters):
                rt.reset_stats()
                rt.execute(schedule)
            elapsed = time.monotonic() - start
            timings[threaded] = elapsed

        seq_ms = timings[False] / n_iters * 1000
        thr_ms = timings[True] / n_iters * 1000
        speedup = timings[False] / timings[True]

        print(f"\n  Sequential: {seq_ms:.3f} ms/iter")
        print(f"  Threaded:   {thr_ms:.3f} ms/iter")
        print(f"  Speedup:    {speedup:.2f}x")

        # Threaded should not be more than 3x slower (thread overhead is bounded)
        assert speedup > 0.3, (
            f"Threaded mode is too slow: {speedup:.2f}x "
            f"(seq={seq_ms:.3f}ms, thr={thr_ms:.3f}ms)"
        )
