#!/usr/bin/env python3
"""Week 3 Demo — MLP with Full Scheduling Pipeline.

This script demonstrates the complete Week 3 pipeline:
1. Build graph (IR)
2. Fusion pass (Week 2)
3. Tiling pass (Week 1)
4. Lowering pass (Week 2)
5. Scheduling pass (Week 3)

Run with:
    python -m examples.mlp_scheduled
"""

from compiler.ir import Graph
from compiler.passes import FusionPass, LoweringPass, TilingPass
from compiler.scheduler import (
    SchedExecute,
    SchedLoad,
    SchedStore,
    Scheduler,
    SRAMConfig,
    format_schedule,
    format_stats,
)


def build_mlp_graph() -> Graph:
    """Build a 3-layer MLP graph for demonstration."""
    g = Graph("mlp_3layer")
    
    # Input: batch=128, features=128
    x = g.input("x", (128, 128))
    
    # Layer 1: 128x128 @ 128x64 -> 128x64, with ReLU
    w1 = g.param("w1", (128, 64))
    h1 = g.matmul(x, w1)
    a1 = g.relu(h1)
    
    # Layer 2: 128x64 @ 64x32 -> 128x32, with ReLU  
    w2 = g.param("w2", (64, 32))
    h2 = g.matmul(a1, w2)
    a2 = g.relu(h2)
    
    # Layer 3: 128x32 @ 32x32 -> 128x32 (no ReLU, output layer)
    w3 = g.param("w3", (32, 32))
    out = g.matmul(a2, w3)
    
    return g


def main():
    print("=" * 70)
    print("Week 3 Demo — MLP Scheduling Pipeline")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Build Graph
    # -------------------------------------------------------------------------
    print("\n[1] Building MLP graph...")
    g = build_mlp_graph()
    print(f"    Original graph: {len(g.ops)} ops")
    print(f"    Tensors: {len(g.tensors)}")
    
    # -------------------------------------------------------------------------
    # Step 2: Fusion Pass
    # -------------------------------------------------------------------------
    print("\n[2] Running Fusion Pass...")
    fusion = FusionPass()
    fusion.run(g)
    print(f"    After fusion: {len(g.ops)} ops")
    
    # Show which ops remain
    for op in g.ops:
        print(f"      - {op.name} ({op.__class__.__name__})")
    
    # -------------------------------------------------------------------------
    # Step 3: Tiling Pass
    # -------------------------------------------------------------------------
    print("\n[3] Running Tiling Pass...")
    tiling = TilingPass()
    tiling.run(g)
    print(f"    Tiling validated: {g.attrs.get('tiling', {}).get('validated', False)}")
    
    # -------------------------------------------------------------------------
    # Step 4: Lowering Pass
    # -------------------------------------------------------------------------
    print("\n[4] Running Lowering Pass...")
    lowering = LoweringPass()
    uops = lowering.run(g)
    print(f"    Generated {len(uops)} micro-ops (uOps)")
    
    # -------------------------------------------------------------------------
    # Step 5: Scheduling Pass (the Week 3 highlight!)
    # -------------------------------------------------------------------------
    print("\n[5] Running Scheduling Pass...")
    
    # Configure SRAM (256 KiB simulated on-chip memory)
    sram_size_kb = 256
    config = SRAMConfig(total_bytes=sram_size_kb * 1024)
    print(f"    SRAM size: {sram_size_kb} KiB")
    
    # Run scheduler (single-buffer mode first)
    scheduler = Scheduler(config=config, double_buffer=False)
    schedule, stats = scheduler.run_on_graph(g)
    
    print("\n" + "=" * 70)
    print("Schedule Statistics (Single Buffer)")
    print("=" * 70)
    print(format_stats(stats))
    
    # -------------------------------------------------------------------------
    # Step 6: Double Buffering Comparison
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Double Buffer Mode Comparison")
    print("-" * 70)
    
    scheduler_db = Scheduler(config=config, double_buffer=True)
    schedule_db, stats_db = scheduler_db.run(uops)
    
    print(format_stats(stats_db))
    
    # -------------------------------------------------------------------------
    # Show Sample Schedule
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Sample Schedule (first 15 operations)")
    print("=" * 70)
    print(format_schedule(schedule, max_lines=15))
    
    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    # Calculate reuse ratio
    total_load_requests = stats.loads_emitted + stats.loads_eliminated
    reuse_pct = 100.0 * stats.loads_eliminated / total_load_requests if total_load_requests > 0 else 0
    
    print(f"  Graph ops:          {len(g.ops)}")
    print(f"  Lowered uOps:       {len(uops)}")
    print(f"  Scheduled ops:      {stats.sched_ops}")
    print(f"  Load reuse:         {stats.loads_eliminated}/{total_load_requests} ({reuse_pct:.1f}%)")
    print(f"  Peak SRAM:          {stats.peak_sram_bytes / 1024:.1f} KiB / {sram_size_kb} KiB")
    print(f"  Final live bytes:   {stats.final_live_bytes}")
    
    # Verify invariants
    print("\n  Invariants:")
    print(f"    ✓ Final live bytes = 0: {'PASS' if stats.final_live_bytes == 0 else 'FAIL'}")
    print(f"    ✓ Load reuse > 0:       {'PASS' if stats.loads_eliminated > 0 else 'FAIL'}")
    print(f"    ✓ Peak < SRAM size:     {'PASS' if stats.peak_sram_bytes < config.total_bytes else 'FAIL'}")
    
    # Check graph attrs were populated
    has_schedule = "schedule" in g.attrs
    has_stats = "schedule_stats" in g.attrs
    print(f"    ✓ Graph attrs set:      {'PASS' if has_schedule and has_stats else 'FAIL'}")
    
    print("\n" + "=" * 70)
    print("Week 3 Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
