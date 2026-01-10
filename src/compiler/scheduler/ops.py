"""Schedule IR — Physical execution schedule with concrete SRAM addresses.

This module defines the scheduled operations that result from the scheduling
pass. Unlike the logical uOps, these ops carry concrete SRAM addresses and
are directly consumable by the runtime engine.

Schedule ops are "flat" — list order defines execution order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# Schedule Operation IR
# =============================================================================


@dataclass(slots=True)
class SchedOp:
    """Base class for all scheduled operations.
    
    All schedule ops carry an optional `buffer` field to support double
    buffering. When `buffer` is None, single-buffering is assumed.
    """
    
    buffer: int | None = field(default=None, kw_only=True)


@dataclass(slots=True)
class SchedLoad(SchedOp):
    """Load a tile from backing memory into SRAM.
    
    Attributes:
        tensor: Name of the source tensor.
        coord: Tile coordinates (e.g., (m, k) for A tiles, (k, n) for B tiles).
        dst_addr: Destination address in SRAM (bytes).
        bytes: Size of the tile in bytes.
        buffer: Optional buffer id for double buffering.
    """
    
    tensor: str
    coord: tuple[int, ...]
    dst_addr: int
    bytes: int
    buffer: int | None = field(default=None, kw_only=True)
    
    def __repr__(self) -> str:
        buf = f" [buf={self.buffer}]" if self.buffer is not None else ""
        return f"LOAD {self.tensor}{self.coord} -> @0x{self.dst_addr:04X} ({self.bytes}B){buf}"


@dataclass(slots=True)
class SchedExecute(SchedOp):
    """Execute a tile-level matrix multiplication accumulation.
    
    Performs C[m,n] += A[m,k] @ B[k,n] on 32x32 tiles.
    
    Attributes:
        m: Output tile row index.
        n: Output tile column index.
        k: Reduction dimension tile index.
        a_addr: SRAM address of A tile (read).
        b_addr: SRAM address of B tile (read).
        acc_addr: SRAM address of accumulator tile (read/write).
        buffer: Optional buffer id for double buffering.
    
    Note:
        acc_addr is both read and written — the accumulator is updated in place.
    """
    
    m: int
    n: int
    k: int
    a_addr: int
    b_addr: int
    acc_addr: int
    buffer: int | None = field(default=None, kw_only=True)
    
    def __repr__(self) -> str:
        buf = f" [buf={self.buffer}]" if self.buffer is not None else ""
        return (
            f"EXEC ({self.m},{self.n},{self.k}) "
            f"A=@0x{self.a_addr:04X} B=@0x{self.b_addr:04X} "
            f"ACC=@0x{self.acc_addr:04X}{buf}"
        )


@dataclass(slots=True)
class SchedStore(SchedOp):
    """Store a tile from SRAM back to backing memory.
    
    Attributes:
        tensor: Name of the destination tensor.
        coord: Tile coordinates (e.g., (m, n) for output tiles).
        src_addr: Source address in SRAM (bytes).
        bytes: Size of the tile in bytes.
        activation: Optional activation function to apply on store (e.g., "relu").
    """
    
    tensor: str
    coord: tuple[int, ...]
    src_addr: int
    bytes: int
    activation: str | None = None
    
    def __repr__(self) -> str:
        act = f" -> {self.activation.upper()}" if self.activation else ""
        return f"STORE @0x{self.src_addr:04X} -> {self.tensor}{self.coord} ({self.bytes}B){act}"


# =============================================================================
# Schedule Statistics
# =============================================================================


@dataclass(slots=True)
class ScheduleStats:
    """Statistics collected during scheduling.
    
    Attributes:
        sched_ops: Total number of scheduled operations emitted.
        loads_emitted: Number of LOAD operations emitted.
        loads_eliminated: Number of redundant loads avoided via tile reuse.
        peak_sram_bytes: High-water mark of SRAM usage.
        final_live_bytes: Bytes still allocated at schedule end (should be 0).
    """
    
    sched_ops: int = 0
    loads_emitted: int = 0
    loads_eliminated: int = 0
    peak_sram_bytes: int = 0
    final_live_bytes: int = 0
    
    def as_dict(self) -> dict[str, int]:
        """Return stats as a plain dict for graph.attrs storage."""
        return {
            "sched_ops": self.sched_ops,
            "loads_emitted": self.loads_emitted,
            "loads_eliminated": self.loads_eliminated,
            "peak_sram_bytes": self.peak_sram_bytes,
            "final_live_bytes": self.final_live_bytes,
        }


# =============================================================================
# Pretty Printing Utilities
# =============================================================================


def format_schedule(
    schedule: Sequence[SchedOp],
    *,
    max_lines: int | None = None,
    indent: str = "  ",
) -> str:
    """Format a schedule as a human-readable string.
    
    Args:
        schedule: Sequence of scheduled operations.
        max_lines: Maximum number of ops to print. None for all.
        indent: Indentation prefix for each line.
    
    Returns:
        Formatted multi-line string.
    
    Example output:
        LOAD x(0, 0) -> @0x0000 (4096B)
        LOAD w1(0, 0) -> @0x1000 (4096B)
        EXEC (0,0,0) A=@0x0000 B=@0x1000 ACC=@0x2000
        ...
    """
    lines: list[str] = []
    ops_to_show = schedule[:max_lines] if max_lines else schedule
    
    for op in ops_to_show:
        lines.append(f"{indent}{op!r}")
    
    if max_lines and len(schedule) > max_lines:
        remaining = len(schedule) - max_lines
        lines.append(f"{indent}... ({remaining} more ops)")
    
    return "\n".join(lines)


def format_stats(stats: ScheduleStats, *, indent: str = "  ") -> str:
    """Format schedule statistics as a human-readable string.
    
    Args:
        stats: The schedule statistics to format.
        indent: Indentation prefix for each line.
    
    Returns:
        Formatted multi-line string.
    """
    reuse_pct = 0.0
    total_load_requests = stats.loads_emitted + stats.loads_eliminated
    if total_load_requests > 0:
        reuse_pct = 100.0 * stats.loads_eliminated / total_load_requests
    
    return "\n".join([
        f"{indent}Total scheduled ops: {stats.sched_ops}",
        f"{indent}Loads emitted:       {stats.loads_emitted}",
        f"{indent}Loads eliminated:    {stats.loads_eliminated} ({reuse_pct:.1f}% reuse)",
        f"{indent}Peak SRAM usage:     {stats.peak_sram_bytes:,} bytes "
        f"({stats.peak_sram_bytes / 1024:.1f} KiB)",
        f"{indent}Final live bytes:    {stats.final_live_bytes}",
    ])


# Type alias for tile keys used in residency tracking
TileKey = tuple[str, tuple[int, ...]]
