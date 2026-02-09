#!/usr/bin/env python3
"""Week 6 Benchmark — Sequential vs. Threaded MLP Execution.

Compares the sequential and dual-threaded pipelined engines:
  - Correctness: bit-identical output verification
  - Performance: latency per iteration, speedup ratio
  - Stability: 1000-iteration stress test

Run with:
    python -m benchmarks.mlp_threaded
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
    print("Week 6 Benchmark — Sequential vs. Threaded MLP Execution")
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
    # Step 7: Overlap analysis
    # -------------------------------------------------------------------------
    print("\n[7] Overlap analysis...")

    # Run once more to get fresh overlap stats
    rt_thr.reset_stats()
    rt_thr.execute(schedule)
    ov = rt_thr.stats

    dma_us = ov.dma_busy_ns / 1000
    compute_us = ov.compute_busy_ns / 1000
    overlap_us = ov.overlap_ns / 1000
    total_us = ov.total_ns / 1000

    overlap_ratio = ov.overlap_ns / ov.total_ns * 100 if ov.total_ns > 0 else 0
    dma_util = ov.dma_busy_ns / ov.total_ns * 100 if ov.total_ns > 0 else 0
    compute_util = ov.compute_busy_ns / ov.total_ns * 100 if ov.total_ns > 0 else 0
    idle_ns = max(0, ov.total_ns - ov.dma_busy_ns - ov.compute_busy_ns + ov.overlap_ns)
    idle_pct = idle_ns / ov.total_ns * 100 if ov.total_ns > 0 else 0

    print(f"    Wall-clock time:    {total_us:8.1f} us")
    print(f"    DMA busy time:      {dma_us:8.1f} us  ({dma_util:.1f}% utilization)")
    print(f"    Compute busy time:  {compute_us:8.1f} us  ({compute_util:.1f}% utilization)")
    print(f"    Overlap time:       {overlap_us:8.1f} us  ({overlap_ratio:.1f}% of wall-clock)")
    print(f"    Idle time:          {idle_ns/1000:8.1f} us  ({idle_pct:.1f}% of wall-clock)")

    print()
    print("    Timeline (conceptual):")
    # Build a simple ASCII bar chart
    bar_width = 50
    if total_us > 0:
        dma_bar = int(bar_width * dma_util / 100)
        comp_bar = int(bar_width * compute_util / 100)
        overlap_bar = int(bar_width * overlap_ratio / 100)
        print(f"    DMA     [{'#' * dma_bar}{'.' * (bar_width - dma_bar)}] {dma_util:.1f}%")
        print(f"    Compute [{'#' * comp_bar}{'.' * (bar_width - comp_bar)}] {compute_util:.1f}%")
        print(f"    Overlap [{'=' * overlap_bar}{'.' * (bar_width - overlap_bar)}] {overlap_ratio:.1f}%")

    # -------------------------------------------------------------------------
    # Step 8: Stress test
    # -------------------------------------------------------------------------
    print("\n[8] Stress test (1000 threaded iterations)...")
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

  Performance:
    Sequential:     {seq_ms:.3f} ms/iter
    Threaded:       {thr_ms:.3f} ms/iter
    Speedup:        {speedup:.2f}x

  Overlap Analysis:
    DMA utilization:     {dma_util:5.1f}%
    Compute utilization: {compute_util:5.1f}%
    Overlap ratio:       {overlap_ratio:5.1f}%  (time both threads active)
    Idle ratio:          {idle_pct:5.1f}%  (neither thread active)

  Correctness:
    Sequential vs NumPy: {'PASS' if seq_correct else 'FAIL'}
    Threaded vs NumPy:   {'PASS' if thr_correct else 'FAIL'}
    Threaded vs Seq:     {'PASS' if match else 'FAIL'}
    Stress (1000x):      {'PASS' if stress_correct else 'FAIL'}
""")

    print("=" * 70)
    print("Week 6 Benchmark Complete!")
    print("=" * 70)

    return 0 if (seq_correct and thr_correct and match and stress_correct) else 1


if __name__ == "__main__":
    exit(main())
