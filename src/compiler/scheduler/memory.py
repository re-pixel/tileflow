"""Week 3 VirtualSRAMArena — Deterministic SRAM allocator for tile scheduling.

This module provides a simple, deterministic memory allocator that simulates
a fixed-size SRAM arena. It is used by the scheduler to assign concrete
addresses to tiles during schedule generation.

Key design principles:
- Determinism: allocation order and addresses are reproducible.
- Debuggability: OOM errors include detailed diagnostics.
- Simplicity: first-fit free-list with alignment, no fancy algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class SRAMConfig:
    """Configuration for the virtual SRAM arena.
    
    Attributes:
        total_bytes: Total size of the SRAM in bytes.
        alignment: Required alignment for allocations in bytes.
                   Must be a power of 2. Default is 64 (cache line).
    """
    
    total_bytes: int
    alignment: int = 64
    
    def __post_init__(self) -> None:
        if self.total_bytes <= 0:
            raise ValueError(f"total_bytes must be positive, got {self.total_bytes}")
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValueError(
                f"alignment must be a positive power of 2, got {self.alignment}"
            )


# =============================================================================
# Tile Size Helpers
# =============================================================================


def tile_bytes(tile_m: int = 32, tile_n: int = 32, dtype_bytes: int = 4) -> int:
    """Compute the size of a tile in bytes.
    
    Args:
        tile_m: Tile height (default 32).
        tile_n: Tile width (default 32).
        dtype_bytes: Bytes per element (default 4 for FP32).
    
    Returns:
        Total bytes for the tile.
    
    Example:
        >>> tile_bytes()  # 32x32 FP32
        4096
    """
    return tile_m * tile_n * dtype_bytes


# Default tile size for 32x32 FP32 tiles
TILE_BYTES = tile_bytes()  # 4096 bytes = 4 KiB


# =============================================================================
# Allocation Tracking
# =============================================================================


class Allocation(NamedTuple):
    """Record of a single allocation in the arena."""
    
    addr: int
    size: int
    tag: str


class FreeBlock(NamedTuple):
    """A contiguous free region in the arena."""
    
    addr: int
    size: int


# =============================================================================
# Exceptions
# =============================================================================


class SRAMOutOfMemoryError(Exception):
    """Raised when SRAM allocation fails due to insufficient space."""
    
    pass


# =============================================================================
# Virtual SRAM Arena
# =============================================================================


@dataclass
class VirtualSRAMArena:
    """Deterministic first-fit allocator for simulated SRAM.
    
    This allocator maintains a sorted free-list and uses first-fit allocation
    with configurable alignment. It tracks peak and live memory usage for
    scheduling diagnostics.
    
    Attributes:
        config: SRAM configuration (size, alignment).
        peak_bytes: High-water mark of allocated bytes.
        live_bytes: Currently allocated bytes.
    
    Example:
        >>> config = SRAMConfig(total_bytes=1024 * 1024)  # 1 MiB
        >>> arena = VirtualSRAMArena(config)
        >>> addr = arena.alloc(4096, tag="A(0,0)")
        >>> arena.free(addr)
    """
    
    config: SRAMConfig
    
    # Internal state
    _allocations: dict[int, Allocation] = field(default_factory=dict, repr=False)
    _free_list: list[FreeBlock] = field(default_factory=list, repr=False)
    _peak_bytes: int = field(default=0, repr=False)
    _live_bytes: int = field(default=0, repr=False)
    _alloc_order: list[int] = field(default_factory=list, repr=False)  # LRU tracking
    
    def __post_init__(self) -> None:
        """Initialize the arena with a single free block spanning all memory."""
        self._free_list = [FreeBlock(addr=0, size=self.config.total_bytes)]
    
    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------
    
    @property
    def peak_bytes(self) -> int:
        """High-water mark of allocated bytes."""
        return self._peak_bytes
    
    @property
    def live_bytes(self) -> int:
        """Currently allocated bytes."""
        return self._live_bytes
    
    @property
    def free_bytes(self) -> int:
        """Currently free bytes (may be fragmented)."""
        return self.config.total_bytes - self._live_bytes
    
    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------
    
    def alloc(self, size: int, tag: str) -> int:
        """Allocate a block of memory.
        
        Args:
            size: Number of bytes to allocate (will be aligned up).
            tag: Human-readable identifier for debugging (e.g., "A(0,0)").
        
        Returns:
            Base address of the allocated block (aligned).
        
        Raises:
            SRAMOutOfMemoryError: If no suitable free block exists.
        """
        if size <= 0:
            raise ValueError(f"Allocation size must be positive, got {size}")
        
        aligned_size = self._align_up(size)
        
        # First-fit search
        for i, block in enumerate(self._free_list):
            # Check if block can accommodate aligned allocation
            aligned_addr = self._align_up(block.addr)
            padding = aligned_addr - block.addr
            required = padding + aligned_size
            
            if block.size >= required:
                return self._allocate_from_block(i, block, aligned_addr, aligned_size, tag)
        
        # Allocation failed — build detailed error message
        self._raise_oom_error(size, aligned_size, tag)
    
    def free(self, addr: int) -> None:
        """Free a previously allocated block.
        
        Args:
            addr: Base address returned by alloc().
        
        Raises:
            KeyError: If addr was not allocated or already freed.
        """
        if addr not in self._allocations:
            raise KeyError(
                f"Cannot free address 0x{addr:04X}: not allocated or already freed"
            )
        
        alloc = self._allocations.pop(addr)
        self._live_bytes -= alloc.size
        
        # Remove from LRU tracking
        if addr in self._alloc_order:
            self._alloc_order.remove(addr)
        
        # Insert freed block back into free list (sorted by address)
        new_block = FreeBlock(addr=alloc.addr, size=alloc.size)
        self._insert_and_coalesce(new_block)
    
    def reset(self) -> None:
        """Reset the arena to initial state, freeing all allocations.
        
        Preserves peak_bytes for post-reset inspection.
        """
        self._allocations.clear()
        self._free_list = [FreeBlock(addr=0, size=self.config.total_bytes)]
        self._live_bytes = 0
        self._alloc_order.clear()
    
    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    
    def get_allocations(self) -> list[Allocation]:
        """Return a list of all current allocations, sorted by address."""
        return sorted(self._allocations.values(), key=lambda a: a.addr)
    
    def get_lru_operand_addr(self) -> int | None:
        """Return the address of the least recently used allocation, or None.
        
        Used by the scheduler for LRU eviction policy.
        """
        if not self._alloc_order:
            return None
        return self._alloc_order[0]
    
    def format_state(self, *, max_allocs: int = 10) -> str:
        """Format arena state for debugging.
        
        Args:
            max_allocs: Maximum number of allocations to show.
        
        Returns:
            Multi-line string describing arena state.
        """
        lines = [
            f"VirtualSRAMArena:",
            f"  Total:     {self.config.total_bytes:,} bytes",
            f"  Live:      {self._live_bytes:,} bytes",
            f"  Peak:      {self._peak_bytes:,} bytes",
            f"  Free:      {self.free_bytes:,} bytes",
            f"  Alignment: {self.config.alignment} bytes",
        ]
        
        allocations = self.get_allocations()
        if allocations:
            lines.append(f"  Allocations ({len(allocations)} total):")
            for alloc in allocations[:max_allocs]:
                lines.append(
                    f"    @0x{alloc.addr:04X}: {alloc.size:,} bytes [{alloc.tag}]"
                )
            if len(allocations) > max_allocs:
                lines.append(f"    ... ({len(allocations) - max_allocs} more)")
        
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------
    
    def _align_up(self, value: int) -> int:
        """Round value up to the next aligned boundary."""
        alignment = self.config.alignment
        return (value + alignment - 1) & ~(alignment - 1)
    
    def _allocate_from_block(
        self,
        block_idx: int,
        block: FreeBlock,
        aligned_addr: int,
        aligned_size: int,
        tag: str,
    ) -> int:
        """Perform allocation from a specific free block."""
        # Remove the block from free list
        self._free_list.pop(block_idx)
        
        # Handle any padding before the aligned address
        if aligned_addr > block.addr:
            prefix = FreeBlock(addr=block.addr, size=aligned_addr - block.addr)
            self._insert_free_block(prefix)
        
        # Handle any remainder after the allocation
        end_addr = aligned_addr + aligned_size
        block_end = block.addr + block.size
        if end_addr < block_end:
            suffix = FreeBlock(addr=end_addr, size=block_end - end_addr)
            self._insert_free_block(suffix)
        
        # Record the allocation
        alloc = Allocation(addr=aligned_addr, size=aligned_size, tag=tag)
        self._allocations[aligned_addr] = alloc
        self._live_bytes += aligned_size
        self._peak_bytes = max(self._peak_bytes, self._live_bytes)
        
        # Track allocation order for LRU
        self._alloc_order.append(aligned_addr)
        
        return aligned_addr
    
    def _insert_free_block(self, block: FreeBlock) -> None:
        """Insert a free block into the sorted free list (no coalescing)."""
        # Binary search for insertion point
        lo, hi = 0, len(self._free_list)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._free_list[mid].addr < block.addr:
                lo = mid + 1
            else:
                hi = mid
        self._free_list.insert(lo, block)
    
    def _insert_and_coalesce(self, block: FreeBlock) -> None:
        """Insert a free block and merge with adjacent free blocks."""
        # Find insertion point
        lo, hi = 0, len(self._free_list)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._free_list[mid].addr < block.addr:
                lo = mid + 1
            else:
                hi = mid
        
        # Check if we can merge with predecessor
        merged = block
        if lo > 0:
            prev = self._free_list[lo - 1]
            if prev.addr + prev.size == merged.addr:
                merged = FreeBlock(addr=prev.addr, size=prev.size + merged.size)
                self._free_list.pop(lo - 1)
                lo -= 1
        
        # Check if we can merge with successor
        if lo < len(self._free_list):
            next_block = self._free_list[lo]
            if merged.addr + merged.size == next_block.addr:
                merged = FreeBlock(addr=merged.addr, size=merged.size + next_block.size)
                self._free_list.pop(lo)
        
        # Insert the (possibly merged) block
        self._free_list.insert(lo, merged)
    
    def _raise_oom_error(self, size: int, aligned_size: int, tag: str) -> None:
        """Raise OOM error with detailed diagnostics."""
        allocations = self.get_allocations()
        
        lines = [
            f"SRAM out of memory: cannot allocate {size} bytes "
            f"(aligned: {aligned_size}) for '{tag}'",
            f"",
            f"Arena state:",
            f"  Total capacity:  {self.config.total_bytes:,} bytes",
            f"  Currently live:  {self._live_bytes:,} bytes",
            f"  Currently free:  {self.free_bytes:,} bytes (possibly fragmented)",
            f"",
        ]
        
        if allocations:
            lines.append(f"Top allocations ({min(len(allocations), 5)} of {len(allocations)}):")
            for alloc in sorted(allocations, key=lambda a: -a.size)[:5]:
                lines.append(
                    f"  @0x{alloc.addr:04X}: {alloc.size:,} bytes [{alloc.tag}]"
                )
        
        # Show fragmentation info
        if self._free_list:
            lines.append(f"")
            lines.append(f"Free blocks ({len(self._free_list)}):")
            for fb in self._free_list[:5]:
                lines.append(f"  @0x{fb.addr:04X}: {fb.size:,} bytes")
            if len(self._free_list) > 5:
                lines.append(f"  ... ({len(self._free_list) - 5} more)")
        
        raise SRAMOutOfMemoryError("\n".join(lines))
