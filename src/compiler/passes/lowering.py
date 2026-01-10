from __future__ import annotations

from dataclasses import dataclass

from compiler.ir import FusedMatMulReLU, Graph, MatMul


@dataclass
class UOp:
    pass


@dataclass
class UOpLoad(UOp):
    tensor: str
    coord: tuple[int, ...]
    
    def __repr__(self) -> str:
        return f"LOAD {self.tensor}{self.coord}"


@dataclass
class UOpExecute(UOp):
    m: int
    n: int
    k: int
    
    def __repr__(self) -> str:
        return f"EXEC ({self.m},{self.n},{self.k})"


@dataclass
class UOpStore(UOp):
    tensor: str
    coord: tuple[int, ...]
    activation: str | None = None
    
    def __repr__(self) -> str:
        act = f" -> {self.activation}" if self.activation else ""
        return f"STORE {self.tensor}{self.coord}{act}"


@dataclass
class LoweringPass:
    """Lowers a Tiled Graph into a sequence of micro-ops (uOps)."""
    
    def run(self, graph: Graph) -> list[UOp]:
        all_uops: list[UOp] = []
        
        # Iterate operations in topological order (creation order).
        for op in graph.ops:
            if isinstance(op, (MatMul, FusedMatMulReLU)):
                all_uops.extend(self._lower_matmul(op))
                
        graph.attrs["lowered"] = all_uops
        return all_uops
        
    def _lower_matmul(self, op: MatMul) -> list[UOp]:
        info = op.attrs.get("matmul")
        if not info:
            raise ValueError(f"Op {op.name} has no tiling metadata. Run tiling pass first.")
            
        tm = info["tiles"]["m"]
        tn = info["tiles"]["n"]
        tk = info["tiles"]["k"]
        
        inp_a = op.inputs[0].name
        inp_b = op.inputs[1].name
        out_name = op.output.name if op.output else "unknown"
        
        activation = "relu" if isinstance(op, FusedMatMulReLU) else None
        
        uops: list[UOp] = []
        
        # Standard loop order: M, N, K
        for m in range(tm):
            for n in range(tn):
                for k in range(tk):
                    # Emit unoptimized load-execute sequence. 
                    # Redundant loads are removed during the scheduling phase.
                    uops.append(UOpLoad(inp_a, (m, k)))
                    uops.append(UOpLoad(inp_b, (k, n)))
                    uops.append(UOpExecute(m, n, k))
                
                # After reduction loop (K), store result
                uops.append(UOpStore(out_name, (m, n), activation=activation))
                
        return uops
