"""Python wrapper for the C++ runtime.

This module provides a high-level Python interface that bridges the Python
scheduler output with the C++ execution engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from compiler.ir import Graph
    from compiler.scheduler.ops import SchedOp as PySchedOp

# Import C++ module (built via setup.py)
try:
    import mini_runtime as _rt
except ImportError as e:
    raise ImportError(
        "mini_runtime C++ extension not found. "
        "Build with: pip install -e ."
    ) from e


__all__ = ["Runtime", "TILE_DIM", "TILE_BYTES"]

# Re-export constants from C++
TILE_DIM = _rt.TILE_DIM
TILE_BYTES = _rt.TILE_BYTES


class Runtime:
    """High-level Python interface to the C++ execution engine.
    
    This class wraps the C++ Engine and handles:
    - Tensor registration and data transfer
    - Schedule conversion from Python to C++ format
    - Result retrieval
    
    Example:
        >>> rt = Runtime(sram_bytes=256 * 1024)
        >>> rt.register_tensor("A", (32, 32))
        >>> rt.set_tensor("A", np.random.randn(32, 32).astype(np.float32))
        >>> rt.execute(schedule)
        >>> result = rt.get_tensor("C", (32, 32))
    """
    
    def __init__(self, sram_bytes: int = 256 * 1024) -> None:
        """Create a runtime with the specified SRAM size.
        
        Args:
            sram_bytes: Size of simulated SRAM in bytes. Default 256 KiB.
        """
        config = _rt.EngineConfig()
        config.sram_bytes = sram_bytes
        self._engine = _rt.Engine(config)
        self._tensor_ids: dict[str, int] = {}
        self._tensor_shapes: dict[str, tuple[int, int]] = {}
    
    def register_tensor(self, name: str, shape: tuple[int, int]) -> int:
        """Register a tensor for use in execution.
        
        Args:
            name: Unique tensor name (must match names in schedule).
            shape: Tensor shape as (rows, cols).
        
        Returns:
            Tensor ID assigned by the engine.
        """
        tid = self._engine.register_tensor(name, shape[0], shape[1])
        self._tensor_ids[name] = tid
        self._tensor_shapes[name] = shape
        return tid
    
    def set_tensor(self, name: str, data: np.ndarray) -> None:
        """Set tensor data from a numpy array.
        
        Args:
            name: Tensor name (must be registered).
            data: 2D numpy array (will be converted to float32).
        """
        if name not in self._tensor_ids:
            raise ValueError(f"Tensor '{name}' not registered")
        
        # Ensure correct dtype and layout
        arr = np.ascontiguousarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.ndim}D")
        
        expected_shape = self._tensor_shapes[name]
        if arr.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for '{name}': "
                f"expected {expected_shape}, got {arr.shape}"
            )
        
        self._engine.set_tensor(name, arr)
    
    def get_tensor(self, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
        """Get tensor data as a numpy array.
        
        Args:
            name: Tensor name.
            shape: Shape to retrieve. If None, uses registered shape.
        
        Returns:
            2D numpy array with tensor data.
        """
        if name not in self._tensor_ids:
            raise ValueError(f"Tensor '{name}' not registered")
        
        if shape is None:
            shape = self._tensor_shapes[name]
        
        return self._engine.get_tensor(name, shape[0], shape[1])
    
    def execute(self, schedule: Sequence[PySchedOp]) -> None:
        """Execute a schedule.
        
        Args:
            schedule: List of Python schedule operations (SchedLoad, SchedExecute, SchedStore).
        """
        cpp_schedule = self._convert_schedule(schedule)
        self._engine.execute(cpp_schedule)
    
    def _convert_schedule(self, schedule: Sequence[PySchedOp]) -> list:
        """Convert Python schedule ops to C++ format."""
        from compiler.scheduler.ops import SchedExecute, SchedLoad, SchedStore
        
        cpp_ops = []
        
        for op in schedule:
            if isinstance(op, SchedLoad):
                tensor_id = self._tensor_ids.get(op.tensor)
                if tensor_id is None:
                    raise ValueError(f"Unknown tensor in schedule: '{op.tensor}'")
                
                cpp_ops.append(_rt.SchedLoad(
                    tensor_id,
                    op.coord[0],
                    op.coord[1],
                    op.dst_addr,
                    op.bytes,
                    op.buffer if op.buffer is not None else -1,
                ))
            
            elif isinstance(op, SchedExecute):
                cpp_ops.append(_rt.SchedExecute(
                    op.m,
                    op.n,
                    op.k,
                    op.a_addr,
                    op.b_addr,
                    op.acc_addr,
                    op.buffer if op.buffer is not None else -1,
                ))
            
            elif isinstance(op, SchedStore):
                tensor_id = self._tensor_ids.get(op.tensor)
                if tensor_id is None:
                    raise ValueError(f"Unknown tensor in schedule: '{op.tensor}'")
                
                cpp_ops.append(_rt.SchedStore(
                    tensor_id,
                    op.coord[0],
                    op.coord[1],
                    op.src_addr,
                    op.bytes,
                    op.activation == "relu",
                ))
            
            else:
                raise TypeError(f"Unknown schedule op type: {type(op)}")
        
        return cpp_ops
    
    @property
    def stats(self) -> "_rt.EngineStats":
        """Get execution statistics."""
        return self._engine.stats()
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._engine.reset_stats()
    
    @property
    def sram_bytes(self) -> int:
        """Get SRAM size in bytes."""
        return self._engine.sram_bytes
    
    def register_graph_tensors(self, graph: "Graph") -> None:
        """Register all tensors from a graph.
        
        Args:
            graph: Compiled graph with tensor definitions.
        """
        for tensor in graph.tensors:
            if tensor.name not in self._tensor_ids:
                self.register_tensor(tensor.name, tensor.shape)
