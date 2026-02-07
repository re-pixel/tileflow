#!/usr/bin/env python3
"""Week 6 Demo — Threaded (Pipelined) MLP Execution.

This script demonstrates the Week 6 dual-threaded pipelined engine:
  - DMA thread handles LOAD/STORE operations
  - Compute thread handles EXEC operations (matmul kernels)
  - Both threads run concurrently, overlapping data movement with compute

The script compares sequential vs. threaded execution for correctness
and measures timing to show overlap benefits.

Run with:
    python -m examples.mlp_threaded
"""

import time

import numpy as np

from compiler.ir import Graph
from compiler.passes import FusionPass, LoweringPass, TilingPass
from compiler.runtime import Runtime, TILE_DIM, TILE_BYTES
from compiler.scheduler import (
    Scheduler,
    SRAMConfig,
    format_schedule,
    format_stats,
)


def build_mlp_graph() -> Graph:
    """Build a 3-layer MLP graph."""
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


def numpy_reference(x, w1, w2, w3):
    """NumPy reference for correctness checking."""
    h1 = np.maximum(0, x @ w1)
    h2 = np.maximum(0, h1 @ w2)
    return h2 @ w3


def benchmark(rt, schedule, n_iters=200):
    """Run schedule n_iters times and return average time in ms."""
    # Warm up
    for _ in range(5):
        rt.execute(schedule)

    start = time.perf_counter()
    for _ in range(n_iters):
        rt.reset_stats()
        rt.execute(schedule)
    elapsed = time.perf_counter() - start

    return elapsed / n_iters * 1000  # ms


def main():
    print("=" * 70)
    print("Week 6 Demo — Threaded (Pipelined) MLP Execution")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Compile
    # -------------------------------------------------------------------------
    print("\n[1] Compiling MLP graph...")
    g = build_mlp_graph()
    FusionPass().run(g)
    TilingPass().run(g)
    LoweringPass().run(g)

    sram_kb = 256
    config = SRAMConfig(total_bytes=sram_kb * 1024)
    scheduler = Scheduler(config=config)
    schedule, stats = scheduler.run_on_graph(g)

    print(f"    Ops after fusion: {len(g.ops)}")
    print(f"    Schedule length:  {stats.sched_ops} ops")
    print(f"    Peak SRAM:        {stats.peak_sram_bytes / 1024:.1f} KiB / {sram_kb} KiB")

    # -------------------------------------------------------------------------
    # Step 2: Prepare data
    # -------------------------------------------------------------------------
    print("\n[2] Preparing input data...")
    np.random.seed(42)
    x_data = np.random.randn(128, 128).astype(np.float32) * 0.1
    w1_data = np.random.randn(128, 64).astype(np.float32) * 0.1
    w2_data = np.random.randn(64, 32).astype(np.float32) * 0.1
    w3_data = np.random.randn(32, 32).astype(np.float32) * 0.1

    expected = numpy_reference(x_data, w1_data, w2_data, w3_data)

    # -------------------------------------------------------------------------
    # Step 3: Sequential execution
    # -------------------------------------------------------------------------
    print("\n[3] Sequential execution...")
    rt_seq = Runtime(sram_bytes=config.total_bytes, threaded=False)
    rt_seq.register_graph_tensors(g)
    rt_seq.set_tensor("x", x_data)
    rt_seq.set_tensor("w1", w1_data)
    rt_seq.set_tensor("w2", w2_data)
    rt_seq.set_tensor("w3", w3_data)

    rt_seq.execute(schedule)
    result_seq = rt_seq.get_tensor("out", (128, 32))
    seq_stats = rt_seq.stats

    seq_correct = np.allclose(result_seq, expected, rtol=1e-4, atol=1e-5)
    print(f"    Correctness: {'PASS' if seq_correct else 'FAIL'}")
    print(f"    Stats: LOADs={seq_stats.loads} EXECs={seq_stats.executes} STOREs={seq_stats.stores}")

    # -------------------------------------------------------------------------
    # Step 4: Threaded execution
    # -------------------------------------------------------------------------
    print("\n[4] Threaded (pipelined) execution...")
    rt_thr = Runtime(sram_bytes=config.total_bytes, threaded=True)
    rt_thr.register_graph_tensors(g)
    rt_thr.set_tensor("x", x_data)
    rt_thr.set_tensor("w1", w1_data)
    rt_thr.set_tensor("w2", w2_data)
    rt_thr.set_tensor("w3", w3_data)

    rt_thr.execute(schedule)
    result_thr = rt_thr.get_tensor("out", (128, 32))
    thr_stats = rt_thr.stats

    thr_correct = np.allclose(result_thr, expected, rtol=1e-4, atol=1e-5)
    print(f"    Correctness: {'PASS' if thr_correct else 'FAIL'}")
    print(f"    Stats: LOADs={thr_stats.loads} EXECs={thr_stats.executes} STOREs={thr_stats.stores}")

    # -------------------------------------------------------------------------
    # Step 5: Verify bit-identical
    # -------------------------------------------------------------------------
    print("\n[5] Comparing sequential vs. threaded output...")
    max_diff = np.max(np.abs(result_seq - result_thr))
    match = np.allclose(result_seq, result_thr, rtol=1e-5, atol=1e-6)
    print(f"    Max difference:  {max_diff:.2e}")
    print(f"    Match:           {'PASS' if match else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Step 6: Benchmark
    # -------------------------------------------------------------------------
    print("\n[6] Benchmarking (200 iterations each)...")

    seq_ms = benchmark(rt_seq, schedule, n_iters=200)
    thr_ms = benchmark(rt_thr, schedule, n_iters=200)
    speedup = seq_ms / thr_ms if thr_ms > 0 else float("inf")

    print(f"    Sequential: {seq_ms:.3f} ms/iter")
    print(f"    Threaded:   {thr_ms:.3f} ms/iter")
    print(f"    Speedup:    {speedup:.2f}x")

    # -------------------------------------------------------------------------
    # Step 7: Stress test
    # -------------------------------------------------------------------------
    print("\n[7] Stress test (1000 threaded iterations)...")
    start = time.perf_counter()
    for _ in range(1000):
        rt_thr.reset_stats()
        rt_thr.execute(schedule)
    stress_elapsed = time.perf_counter() - start

    result_stress = rt_thr.get_tensor("out", (128, 32))
    stress_correct = np.allclose(result_stress, expected, rtol=1e-4, atol=1e-5)

    print(f"    Total time:  {stress_elapsed:.2f}s")
    print(f"    Correctness: {'PASS' if stress_correct else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
  Architecture:
    DMA Thread:     LOAD / STORE operations (data movement)
    Compute Thread: EXEC operations (matmul kernels)
    Communication:  Lock-free SPSC ring buffers (16-slot depth)

  Results:
    Sequential:     {seq_ms:.3f} ms/iter
    Threaded:       {thr_ms:.3f} ms/iter
    Speedup:        {speedup:.2f}x
    Correctness:    {'ALL PASS' if (seq_correct and thr_correct and match and stress_correct) else 'SOME FAILURES'}
    Stress (1000x): {'PASS' if stress_correct else 'FAIL'}
""")

    print("=" * 70)
    print("Week 6 Demo Complete!")
    print("=" * 70)

    return 0 if (seq_correct and thr_correct and match and stress_correct) else 1


if __name__ == "__main__":
    exit(main())
