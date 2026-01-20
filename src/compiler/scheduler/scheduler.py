"""Scheduler — Converts logical uOps into scheduled ops with SRAM addresses.

This module implements the static scheduling algorithm that:
1. Allocates SRAM addresses for tile loads, executes, and stores.
2. Tracks tile residency to eliminate redundant loads.
3. Manages accumulator and operand tile lifetimes.
4. Supports optional double buffering for pipelined execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from compiler.scheduler.ops import (
    SchedExecute,
    SchedLoad,
    SchedOp,
    SchedStore,
    ScheduleStats,
    TileKey,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from compiler.ir import Graph
    from compiler.passes.lowering import UOp
    from compiler.scheduler.memory import SRAMConfig, VirtualSRAMArena


class SchedulerError(Exception):
    """Raised when scheduling fails due to resource constraints."""
    
    pass


@dataclass
class Scheduler:
    """Static scheduler that converts logical uOps into scheduled ops with SRAM addresses.
    
    The scheduling algorithm:
    1. Precompute remaining uses for operand tiles (A/B) and accumulators (C).
    2. Allocate SRAM on first reference, skip loads if tile is resident.
    3. Free operand tiles when remaining uses hit 0.
    4. Accumulator tiles are allocated on first EXEC(m,n,*) and freed on STORE(m,n).
    5. If OOM, evict operand tiles using LRU (never evict accumulators).
    
    Attributes:
        config: SRAM configuration.
        double_buffer: Enable double buffering for operand tiles within K-loops.
    
    Invariants:
        - Accumulators are allocated on the first SchedExecute(m,n,*).
        - Accumulators are freed on the matching SchedStore(m,n).
        - Accumulators are never evicted.
    """
    
    config: "SRAMConfig"
    double_buffer: bool = False
    
    def run_on_graph(self, graph: "Graph") -> tuple[list[SchedOp], ScheduleStats]:
        """Run scheduling on a graph that has been lowered.
        
        Reads uOps from graph.attrs["lowered"] and writes results to:
          - graph.attrs["schedule"] = list[SchedOp]
          - graph.attrs["schedule_stats"] = dict
        
        Args:
            graph: A graph that has been through FusionPass, TilingPass, and LoweringPass.
        
        Returns:
            Tuple of (scheduled_ops, stats).
        
        Raises:
            ValueError: If graph has not been lowered.
            SchedulerError: If scheduling fails.
        """
        uops = graph.attrs.get("lowered")
        if uops is None:
            raise ValueError(
                "Graph has not been lowered. Run LoweringPass first. "
                f"Available attrs: {list(graph.attrs.keys())}"
            )
        
        schedule, stats = self.run(uops)
        
        # Store results in graph attrs
        graph.attrs["schedule"] = schedule
        graph.attrs["schedule_stats"] = stats.as_dict()
        
        return schedule, stats
    
    def run(self, uops: Sequence["UOp"]) -> tuple[list[SchedOp], ScheduleStats]:
        """Schedule a sequence of uOps into executable SchedOps.
        
        Args:
            uops: Logical micro-ops from the lowering pass.
        
        Returns:
            Tuple of (scheduled_ops, stats).
        
        Raises:
            SchedulerError: If scheduling fails (e.g., cannot fit in SRAM).
        """
        from compiler.passes.lowering import UOpExecute, UOpLoad, UOpStore
        from compiler.scheduler.memory import (
            SRAMOutOfMemoryError,
            TILE_BYTES,
            VirtualSRAMArena,
        )
        
        # Initialize arena
        arena = VirtualSRAMArena(self.config)
        
        # Precompute liveness
        remaining_uses_operands, remaining_uses_accumulators = self._compute_liveness(uops)
        
        # Residency tracking
        resident: dict[TileKey, int] = {}  # TileKey -> SRAM address
        accumulator_keys: set[TileKey] = set()  # Track which keys are accumulators
        
        # Double buffering state
        current_mn: tuple[int, int] | None = None
        
        # Output
        schedule: list[SchedOp] = []
        stats = ScheduleStats()
        
        # Track current operation's output tensor for multi-op graphs.
        # Lowering emits ops per-matmul in order, so we track which "group" we're in.
        current_op_idx = 0
        op_boundaries = self._find_op_boundaries(uops)
        
        # Track the last two loaded tiles for proper A/B identification in EXEC.
        # Lowering emits: LOAD A[m,k], LOAD B[k,n], EXEC(m,n,k)
        pending_loads: list[TileKey] = []
        
        for uop_idx, uop in enumerate(uops):
            # Update current op index when we cross a boundary
            while current_op_idx < len(op_boundaries) - 1 and uop_idx >= op_boundaries[current_op_idx + 1]:
                current_op_idx += 1
            
            if isinstance(uop, UOpLoad):
                tile_key: TileKey = (uop.tensor, uop.coord)
                
                if tile_key in resident:
                    # Tile already in SRAM — skip load, but still track for EXEC
                    stats.loads_eliminated += 1
                else:
                    # Need to load — possibly evict first
                    addr = self._try_alloc_with_eviction(
                        arena=arena,
                        size=TILE_BYTES,
                        tag=f"{uop.tensor}{uop.coord}",
                        resident=resident,
                        remaining_uses_operands=remaining_uses_operands,
                        accumulator_keys=accumulator_keys,
                    )
                    resident[tile_key] = addr
                    
                    sched_load = SchedLoad(
                        tensor=uop.tensor,
                        coord=uop.coord,
                        dst_addr=addr,
                        bytes=TILE_BYTES,
                        buffer=None,  # Will be backpatched for double buffering
                    )
                    schedule.append(sched_load)
                    stats.loads_emitted += 1
                
                # Track this load for the upcoming EXEC
                pending_loads.append(tile_key)
            
            elif isinstance(uop, UOpExecute):
                # Track (m,n) for double buffering scope
                new_mn = (uop.m, uop.n)
                if new_mn != current_mn:
                    current_mn = new_mn
                
                # Determine buffer for double buffering (within same (m,n) K-loop)
                buffer_id = None
                if self.double_buffer:
                    buffer_id = uop.k % 2
                
                # Get A and B tile keys from the two preceding loads.
                # Lowering guarantees: LOAD A[m,k], LOAD B[k,n], EXEC(m,n,k)
                if len(pending_loads) < 2:
                    raise SchedulerError(
                        f"EXEC({uop.m},{uop.n},{uop.k}): expected 2 preceding LOADs, "
                        f"got {len(pending_loads)}: {pending_loads}"
                    )
                
                # A is first, B is second
                a_key = pending_loads[-2]
                b_key = pending_loads[-1]
                pending_loads.clear()  # Reset for next EXEC
                
                if a_key not in resident or b_key not in resident:
                    raise SchedulerError(
                        f"EXEC({uop.m},{uop.n},{uop.k}): operand tiles not in SRAM. "
                        f"A_key={a_key} (in={a_key in resident}), "
                        f"B_key={b_key} (in={b_key in resident})"
                    )
                
                a_addr = resident[a_key]
                b_addr = resident[b_key]

                
                # Accumulator key: use a canonical format that includes op index
                # to handle multiple matmuls in the same graph
                acc_key: TileKey = (f"_acc_{current_op_idx}", (uop.m, uop.n))
                
                if acc_key not in resident:
                    # First EXEC for this (m,n) — allocate accumulator
                    acc_addr = self._try_alloc_with_eviction(
                        arena=arena,
                        size=TILE_BYTES,
                        tag=f"ACC[{current_op_idx}]({uop.m},{uop.n})",
                        resident=resident,
                        remaining_uses_operands=remaining_uses_operands,
                        accumulator_keys=accumulator_keys,
                    )
                    resident[acc_key] = acc_addr
                    accumulator_keys.add(acc_key)
                
                acc_addr = resident[acc_key]
                
                sched_exec = SchedExecute(
                    m=uop.m,
                    n=uop.n,
                    k=uop.k,
                    a_addr=a_addr,
                    b_addr=b_addr,
                    acc_addr=acc_addr,
                    buffer=buffer_id,
                )
                schedule.append(sched_exec)
                
                # Decrement operand uses and free if done
                self._decrement_and_maybe_free(
                    a_key, remaining_uses_operands, resident, accumulator_keys, arena
                )
                self._decrement_and_maybe_free(
                    b_key, remaining_uses_operands, resident, accumulator_keys, arena
                )
                
                # Update double-buffer assignments for preceding loads if needed
                if self.double_buffer and buffer_id is not None:
                    self._backpatch_load_buffers(schedule, a_key, b_key, buffer_id)
            
            elif isinstance(uop, UOpStore):
                # Find accumulator using the canonical key format
                acc_key = (f"_acc_{current_op_idx}", uop.coord)
                
                if acc_key not in resident:
                    raise SchedulerError(
                        f"STORE {uop.tensor}{uop.coord}: accumulator not found in SRAM. "
                        f"Looking for key {acc_key}, resident accumulators: "
                        f"{[k for k in resident if k in accumulator_keys]}"
                    )
                
                acc_addr = resident[acc_key]
                
                sched_store = SchedStore(
                    tensor=uop.tensor,
                    coord=uop.coord,
                    src_addr=acc_addr,
                    bytes=TILE_BYTES,
                    activation=uop.activation,
                )
                schedule.append(sched_store)
                
                # Free accumulator (per invariant)
                arena.free(acc_addr)
                del resident[acc_key]
                accumulator_keys.discard(acc_key)
        
        # Finalize stats
        stats.sched_ops = len(schedule)
        stats.peak_sram_bytes = arena.peak_bytes
        stats.final_live_bytes = arena.live_bytes
        
        return schedule, stats
    
    def _find_op_boundaries(self, uops: Sequence["UOp"]) -> list[int]:
        """Find indices where a new matmul operation starts.
        
        Lowering emits all uOps for one matmul, then the next.
        A new matmul starts when we see a STORE followed by a LOAD.
        """
        from compiler.passes.lowering import UOpLoad, UOpStore
        
        boundaries = [0]
        prev_was_store = False
        
        for i, uop in enumerate(uops):
            if isinstance(uop, UOpLoad) and prev_was_store:
                boundaries.append(i)
            prev_was_store = isinstance(uop, UOpStore)
        
        return boundaries
    
    def _compute_liveness(
        self, uops: Sequence["UOp"]
    ) -> tuple[dict[TileKey, int], dict[TileKey, int]]:
        """Precompute remaining uses for operand and accumulator tiles.
        
        - Operand tiles (A/B): count each LOAD as one use (consumed by following EXEC).
        - Accumulator tiles (C): count as 1 (survives until STORE).
        """
        from compiler.passes.lowering import UOpExecute, UOpLoad, UOpStore
        
        remaining_uses_operands: dict[TileKey, int] = {}
        remaining_uses_accumulators: dict[TileKey, int] = {}
        
        for uop in uops:
            if isinstance(uop, UOpLoad):
                key: TileKey = (uop.tensor, uop.coord)
                remaining_uses_operands[key] = remaining_uses_operands.get(key, 0) + 1
            elif isinstance(uop, UOpStore):
                # Accumulator: lifetime is 1 (allocated at first EXEC, freed at STORE)
                key = (uop.tensor, uop.coord)
                remaining_uses_accumulators[key] = 1
        
        return remaining_uses_operands, remaining_uses_accumulators
    
    def _try_alloc_with_eviction(
        self,
        arena: "VirtualSRAMArena",
        size: int,
        tag: str,
        resident: dict[TileKey, int],
        remaining_uses_operands: dict[TileKey, int],
        accumulator_keys: set[TileKey],
    ) -> int:
        """Try to allocate, evicting LRU operand tiles if needed."""
        from compiler.scheduler.memory import SRAMOutOfMemoryError
        
        try:
            return arena.alloc(size, tag)
        except SRAMOutOfMemoryError:
            # Evict operand tiles (never accumulators) using LRU
            evicted = False
            
            # Get LRU order from arena
            while True:
                lru_addr = arena.get_lru_operand_addr()
                if lru_addr is None:
                    break
                
                # Find which tile is at this address
                tile_to_evict: TileKey | None = None
                for key, addr in resident.items():
                    if addr == lru_addr and key not in accumulator_keys:
                        tile_to_evict = key
                        break
                
                if tile_to_evict is None:
                    # No evictable operand found at LRU address, try next
                    # This shouldn't happen in normal operation
                    break
                
                # Evict this tile
                arena.free(lru_addr)
                del resident[tile_to_evict]
                evicted = True
                
                # Try allocation again
                try:
                    return arena.alloc(size, tag)
                except SRAMOutOfMemoryError:
                    # Need to evict more
                    continue
            
            # If we still can't allocate, raise detailed error
            if not evicted:
                raise SchedulerError(
                    f"Cannot allocate {size} bytes for '{tag}': "
                    f"no evictable operand tiles (all are accumulators or SRAM is full). "
                    f"Arena state:\n{arena.format_state()}"
                )
            
            # Final attempt
            return arena.alloc(size, tag)
    
    def _decrement_and_maybe_free(
        self,
        key: TileKey,
        remaining_uses: dict[TileKey, int],
        resident: dict[TileKey, int],
        accumulator_keys: set[TileKey],
        arena: "VirtualSRAMArena",
    ) -> None:
        """Decrement operand use count and free if zero."""
        if key in accumulator_keys:
            # Don't decrement accumulators per-EXEC
            return
        
        if key in remaining_uses:
            remaining_uses[key] -= 1
            if remaining_uses[key] <= 0 and key in resident:
                arena.free(resident[key])
                del resident[key]
    
    def _backpatch_load_buffers(
        self,
        schedule: list[SchedOp],
        a_key: TileKey,
        b_key: TileKey,
        buffer_id: int,
    ) -> None:
        """Backpatch buffer IDs on recent LOAD ops for double buffering.
        
        Called after adding a SchedExecute to the schedule. Walks backwards
        from the EXEC to find and update the corresponding LOAD ops.
        """
        # Start from len-2 to skip the EXEC we just added
        for i in range(len(schedule) - 2, -1, -1):
            op = schedule[i]
            if isinstance(op, SchedLoad):
                op_key = (op.tensor, op.coord)
                if op_key == a_key or op_key == b_key:
                    # Replace with updated buffer field
                    schedule[i] = SchedLoad(
                        tensor=op.tensor,
                        coord=op.coord,
                        dst_addr=op.dst_addr,
                        bytes=op.bytes,
                        buffer=buffer_id,
                    )
            elif isinstance(op, SchedExecute):
                # Stop when we hit the previous EXEC
                break
