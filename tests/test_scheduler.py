"""Week 3 Scheduler Tests — VirtualSRAMArena and Scheduler verification.

Tests cover:
1. VirtualSRAMArena basics (alloc/free, alignment, peak tracking, OOM)
2. Intentional OOM with detailed error message
3. Redundant load elimination
4. Liveness-based memory freeing
5. Double buffering annotations
6. End-to-end MLP smoke test with graph integration
"""

import pytest

from compiler.ir import Graph
from compiler.passes import FusionPass, LoweringPass, TilingPass
from compiler.passes.lowering import UOpExecute, UOpLoad, UOpStore
from compiler.scheduler import (
    TILE_BYTES,
    SchedExecute,
    SchedLoad,
    SchedStore,
    Scheduler,
    ScheduleStats,
    SRAMConfig,
    SRAMOutOfMemoryError,
    VirtualSRAMArena,
)


# =============================================================================
# 1. VirtualSRAMArena Basics
# =============================================================================


class TestVirtualSRAMArenaBasics:
    """Test basic allocator functionality."""
    
    def test_alloc_free_roundtrip(self):
        """Allocate and free should work without error."""
        config = SRAMConfig(total_bytes=64 * 1024)
        arena = VirtualSRAMArena(config)
        
        addr = arena.alloc(TILE_BYTES, tag="test_tile")
        assert addr >= 0
        assert arena.live_bytes == TILE_BYTES
        
        arena.free(addr)
        assert arena.live_bytes == 0
    
    def test_alignment_respected(self):
        """Allocations should be aligned to config.alignment."""
        config = SRAMConfig(total_bytes=64 * 1024, alignment=128)
        arena = VirtualSRAMArena(config)
        
        # Allocate several blocks
        addrs = []
        for i in range(5):
            addr = arena.alloc(100, tag=f"block_{i}")  # 100 bytes, not aligned
            addrs.append(addr)
        
        # All addresses should be 128-byte aligned
        for addr in addrs:
            assert addr % 128 == 0, f"Address 0x{addr:04X} not 128-byte aligned"
    
    def test_peak_bytes_tracked(self):
        """Peak bytes should track the high-water mark."""
        config = SRAMConfig(total_bytes=64 * 1024)
        arena = VirtualSRAMArena(config)
        
        # Allocate 3 tiles
        addr1 = arena.alloc(TILE_BYTES, tag="t1")
        addr2 = arena.alloc(TILE_BYTES, tag="t2")
        addr3 = arena.alloc(TILE_BYTES, tag="t3")
        
        assert arena.peak_bytes == 3 * TILE_BYTES
        assert arena.live_bytes == 3 * TILE_BYTES
        
        # Free one
        arena.free(addr2)
        assert arena.live_bytes == 2 * TILE_BYTES
        assert arena.peak_bytes == 3 * TILE_BYTES  # Peak unchanged
        
        # Free all
        arena.free(addr1)
        arena.free(addr3)
        assert arena.live_bytes == 0
        assert arena.peak_bytes == 3 * TILE_BYTES  # Peak still at high-water mark
    
    def test_oom_raises_with_message(self):
        """OOM should raise with a helpful error message."""
        # Tiny SRAM: only 1 tile fits
        config = SRAMConfig(total_bytes=TILE_BYTES)
        arena = VirtualSRAMArena(config)
        
        arena.alloc(TILE_BYTES, tag="first_tile")
        
        with pytest.raises(SRAMOutOfMemoryError) as exc_info:
            arena.alloc(TILE_BYTES, tag="second_tile")
        
        error_msg = str(exc_info.value)
        assert "cannot allocate" in error_msg.lower()
        assert "second_tile" in error_msg
    
    def test_free_nonexistent_raises(self):
        """Freeing an unallocated address should raise KeyError."""
        config = SRAMConfig(total_bytes=64 * 1024)
        arena = VirtualSRAMArena(config)
        
        with pytest.raises(KeyError):
            arena.free(0x1234)
    
    def test_reset_clears_allocations(self):
        """Reset should free all allocations but preserve peak."""
        config = SRAMConfig(total_bytes=64 * 1024)
        arena = VirtualSRAMArena(config)
        
        arena.alloc(TILE_BYTES, tag="t1")
        arena.alloc(TILE_BYTES, tag="t2")
        original_peak = arena.peak_bytes
        
        arena.reset()
        
        assert arena.live_bytes == 0
        assert arena.peak_bytes == original_peak


# =============================================================================
# 1b. Intentional OOM Test (Debuggability)
# =============================================================================


class TestOOMDebugInfo:
    """Test that OOM errors provide detailed diagnostic information."""
    
    def test_oom_shows_capacity_and_live_bytes(self):
        """OOM error should show total capacity and current live bytes."""
        config = SRAMConfig(total_bytes=8 * 1024)  # 8 KiB = 2 tiles
        arena = VirtualSRAMArena(config)
        
        arena.alloc(TILE_BYTES, tag="A(0,0)")
        arena.alloc(TILE_BYTES, tag="B(0,0)")
        
        with pytest.raises(SRAMOutOfMemoryError) as exc_info:
            arena.alloc(TILE_BYTES, tag="C(0,0)")
        
        error_msg = str(exc_info.value)
        
        # Should mention capacity
        assert "8,192" in error_msg or "8192" in error_msg
        
        # Should mention live bytes
        assert "live" in error_msg.lower()
        
        # Should include allocation tags
        assert "A(0,0)" in error_msg
        assert "B(0,0)" in error_msg
    
    def test_oom_includes_top_allocations(self):
        """OOM should list top allocations for debugging."""
        config = SRAMConfig(total_bytes=12 * 1024)  # 3 tiles
        arena = VirtualSRAMArena(config)
        
        arena.alloc(TILE_BYTES, tag="tile_alpha")
        arena.alloc(TILE_BYTES, tag="tile_beta")
        arena.alloc(TILE_BYTES, tag="tile_gamma")
        
        with pytest.raises(SRAMOutOfMemoryError) as exc_info:
            arena.alloc(TILE_BYTES, tag="tile_overflow")
        
        error_msg = str(exc_info.value)
        
        # Should list existing allocations
        assert "tile_alpha" in error_msg or "allocations" in error_msg.lower()


# =============================================================================
# 2. Redundant Load Elimination
# =============================================================================


class TestRedundantLoadElimination:
    """Test that the scheduler eliminates redundant loads."""
    
    def test_same_tile_loaded_once(self):
        """Loading the same tile twice should result in only one SchedLoad."""
        # Create uOps that load the same tile twice
        uops = [
            UOpLoad("x", (0, 0)),
            UOpLoad("w", (0, 0)),
            UOpExecute(0, 0, 0),
            UOpLoad("x", (0, 0)),  # Same as first load - should be eliminated
            UOpLoad("w", (0, 1)),
            UOpExecute(0, 1, 0),
            UOpStore("out", (0, 0)),
            UOpStore("out", (0, 1)),
        ]
        
        config = SRAMConfig(total_bytes=64 * 1024)
        scheduler = Scheduler(config=config, double_buffer=False)
        schedule, stats = scheduler.run(uops)
        
        # Count loads for x(0,0)
        x_loads = [
            op for op in schedule 
            if isinstance(op, SchedLoad) and op.tensor == "x" and op.coord == (0, 0)
        ]
        
        assert len(x_loads) == 1, f"Expected 1 load for x(0,0), got {len(x_loads)}"
        assert stats.loads_eliminated >= 1


# =============================================================================
# 3. Liveness Frees Memory
# =============================================================================


class TestLivenessFreesMemory:
    """Test that tiles are freed when no longer needed."""
    
    def test_final_live_bytes_zero(self):
        """After scheduling completes, all tiles should be freed."""
        # Simple matmul with single output tile
        uops = [
            UOpLoad("a", (0, 0)),
            UOpLoad("b", (0, 0)),
            UOpExecute(0, 0, 0),
            UOpStore("c", (0, 0)),
        ]
        
        config = SRAMConfig(total_bytes=64 * 1024)
        scheduler = Scheduler(config=config)
        schedule, stats = scheduler.run(uops)
        
        assert stats.final_live_bytes == 0
    
    def test_peak_bytes_reasonable(self):
        """Peak bytes should not exceed theoretical maximum for workload."""
        # 2x2 output tiles, K=2 reduction
        uops = []
        for m in range(2):
            for n in range(2):
                for k in range(2):
                    uops.append(UOpLoad("a", (m, k)))
                    uops.append(UOpLoad("b", (k, n)))
                    uops.append(UOpExecute(m, n, k))
                uops.append(UOpStore("c", (m, n)))
        
        config = SRAMConfig(total_bytes=256 * 1024)
        scheduler = Scheduler(config=config)
        schedule, stats = scheduler.run(uops)
        
        # Peak should be reasonable (not accumulating all tiles forever)
        # Theoretical max: a few A tiles + a few B tiles + accumulators
        max_reasonable = 20 * TILE_BYTES  # Very generous upper bound
        assert stats.peak_sram_bytes < max_reasonable
        assert stats.final_live_bytes == 0


# =============================================================================
# 4. Double Buffering Annotations
# =============================================================================


class TestDoubleBuffering:
    """Test double buffering mode."""
    
    def test_alternating_buffer_ids(self):
        """EXEC ops should have alternating buffer IDs within a K-loop."""
        # Single output tile with K=4 reduction
        uops = []
        for k in range(4):
            uops.append(UOpLoad("a", (0, k)))
            uops.append(UOpLoad("b", (k, 0)))
            uops.append(UOpExecute(0, 0, k))
        uops.append(UOpStore("c", (0, 0)))
        
        config = SRAMConfig(total_bytes=64 * 1024)
        scheduler = Scheduler(config=config, double_buffer=True)
        schedule, stats = scheduler.run(uops)
        
        exec_ops = [op for op in schedule if isinstance(op, SchedExecute)]
        
        # Should alternate: 0, 1, 0, 1
        expected_buffers = [0, 1, 0, 1]
        actual_buffers = [op.buffer for op in exec_ops]
        
        assert actual_buffers == expected_buffers, \
            f"Expected {expected_buffers}, got {actual_buffers}"
    
    def test_load_buffer_matches_exec(self):
        """LOAD ops should have the same buffer ID as their corresponding EXEC."""
        uops = []
        for k in range(2):
            uops.append(UOpLoad("a", (0, k)))
            uops.append(UOpLoad("b", (k, 0)))
            uops.append(UOpExecute(0, 0, k))
        uops.append(UOpStore("c", (0, 0)))
        
        config = SRAMConfig(total_bytes=64 * 1024)
        scheduler = Scheduler(config=config, double_buffer=True)
        schedule, stats = scheduler.run(uops)
        
        # Group loads by their following exec
        load_ops = [op for op in schedule if isinstance(op, SchedLoad)]
        exec_ops = [op for op in schedule if isinstance(op, SchedExecute)]
        
        # First two loads should have buffer=0, next two buffer=1
        assert load_ops[0].buffer == 0
        assert load_ops[1].buffer == 0
        assert load_ops[2].buffer == 1
        assert load_ops[3].buffer == 1
    
    def test_buffer_resets_on_mn_change(self):
        """Buffer should reset to 0 when (m,n) changes."""
        # Two output tiles, K=2 each
        uops = []
        for m in range(2):
            for k in range(2):
                uops.append(UOpLoad("a", (m, k)))
                uops.append(UOpLoad("b", (k, 0)))
                uops.append(UOpExecute(m, 0, k))
            uops.append(UOpStore("c", (m, 0)))
        
        config = SRAMConfig(total_bytes=64 * 1024)
        scheduler = Scheduler(config=config, double_buffer=True)
        schedule, stats = scheduler.run(uops)
        
        exec_ops = [op for op in schedule if isinstance(op, SchedExecute)]
        
        # First (m=0): k=0 -> buf=0, k=1 -> buf=1
        # Second (m=1): k=0 -> buf=0 (reset), k=1 -> buf=1
        expected = [0, 1, 0, 1]
        actual = [op.buffer for op in exec_ops]
        
        assert actual == expected


# =============================================================================
# 5. End-to-End MLP Smoke Test
# =============================================================================


class TestEndToEndMLP:
    """End-to-end test with full MLP pipeline."""
    
    def test_mlp_schedule_non_empty(self):
        """Scheduling an MLP should produce a non-empty schedule."""
        g = Graph("test_mlp")
        x = g.input("x", (64, 64))
        w1 = g.param("w1", (64, 32))
        h1 = g.matmul(x, w1)
        a1 = g.relu(h1)
        w2 = g.param("w2", (32, 32))
        out = g.matmul(a1, w2)
        
        FusionPass().run(g)
        TilingPass().run(g)
        LoweringPass().run(g)
        
        config = SRAMConfig(total_bytes=256 * 1024)
        scheduler = Scheduler(config=config)
        schedule, stats = scheduler.run_on_graph(g)
        
        assert len(schedule) > 0
        assert stats.sched_ops > 0
    
    def test_mlp_contains_exec_ops(self):
        """Schedule should contain SchedExecute operations."""
        g = Graph("test_mlp")
        x = g.input("x", (64, 64))
        w = g.param("w", (64, 64))
        out = g.matmul(x, w)
        
        TilingPass().run(g)
        LoweringPass().run(g)
        
        config = SRAMConfig(total_bytes=256 * 1024)
        scheduler = Scheduler(config=config)
        schedule, stats = scheduler.run_on_graph(g)
        
        exec_count = sum(1 for op in schedule if isinstance(op, SchedExecute))
        assert exec_count > 0
    
    def test_load_reuse_happening(self):
        """Scheduler should eliminate some redundant loads."""
        g = Graph("test_mlp")
        x = g.input("x", (64, 64))
        w = g.param("w", (64, 64))
        out = g.matmul(x, w)
        
        TilingPass().run(g)
        uops = LoweringPass().run(g)
        
        config = SRAMConfig(total_bytes=256 * 1024)
        scheduler = Scheduler(config=config)
        schedule, stats = scheduler.run(uops)
        
        # Count raw loads in uOps
        raw_load_count = sum(1 for u in uops if isinstance(u, UOpLoad))
        
        # Scheduled loads should be less due to reuse
        assert stats.loads_emitted < raw_load_count
        assert stats.loads_eliminated > 0
    
    def test_graph_attrs_populated(self):
        """Graph attrs should contain schedule and schedule_stats."""
        g = Graph("test_mlp")
        x = g.input("x", (32, 32))
        w = g.param("w", (32, 32))
        out = g.matmul(x, w)
        
        TilingPass().run(g)
        LoweringPass().run(g)
        
        config = SRAMConfig(total_bytes=64 * 1024)
        scheduler = Scheduler(config=config)
        scheduler.run_on_graph(g)
        
        # Check schedule is stored
        assert "schedule" in g.attrs
        assert len(g.attrs["schedule"]) > 0
        
        # Check stats are stored with required keys
        assert "schedule_stats" in g.attrs
        stats = g.attrs["schedule_stats"]
        
        required_keys = [
            "sched_ops",
            "loads_emitted", 
            "loads_eliminated",
            "peak_sram_bytes",
            "final_live_bytes",
        ]
        for key in required_keys:
            assert key in stats, f"Missing required key: {key}"
