from __future__ import annotations

from dataclasses import dataclass

from compiler.ir import Graph, FusedMatMulReLU, MatMul, ReLU


@dataclass(slots=True)
class FusionPass:
    """Fuses MatMul + ReLU into FusedMatMulReLU."""

    def run(self, graph: Graph) -> None:
        # 1. Identify candidates: (MatMul, ReLU) pairs
        # Condition: MatMul feeds ReLU, and ReLU is the *only* consumer of MatMul.
        candidates: list[tuple[MatMul, ReLU]] = []
        
        for op in graph.ops:
            if not isinstance(op, MatMul):
                continue
            
            # Check output consumers
            out = op.output
            if out is None:
                continue
                
            if len(out.users) == 1:
                user = out.users[0]
                if isinstance(user, ReLU):
                    candidates.append((op, user))

        if not candidates:
            return

        # 2. Apply fusion
        replaced_ops: set[int] = set() # id(op)
        new_ops_map: dict[int, FusedMatMulReLU] = {} # id(matmul) -> fused_op

        for mm, relu in candidates:
            # Create fused op
            # Re-use names if possible or generate new one
            fused_name = f"fused_{mm.name}_{relu.name}"
            fused_op = FusedMatMulReLU(name=fused_name, inputs=list(mm.inputs))
            
            # Wire tensor users/producers
            
            # 1. Inputs to MatMul now feed FusedOp
            for t in mm.inputs:
                # Remove mm from users (handle carefully if t used multiple times by same op? 
                # Our list remove removes first occurrence. Assume inputs unique per op for now or iterate safe)
                if mm in t.users:
                    t.users.remove(mm)
                t.add_user(fused_op)

            # 2. Output of ReLU becomes Output of FusedOp
            relu_out = relu.output
            if relu_out is None:
                continue # Should not happen in valid graph
            
            fused_op.output = relu_out
            relu_out.producer = fused_op

            # 3. The intermediate tensor (mm.output) is effectively dead (orphaned)
            # We don't strictly need to clear its fields, but good practice.
            mm.output.producer = None
            mm.output.users = []

            # Record needed graph updates
            replaced_ops.add(id(mm))
            replaced_ops.add(id(relu))
            new_ops_map[id(mm)] = fused_op

        # 3. Rebuild graph.ops list
        new_ops_list = []
        for op in graph.ops:
            oid = id(op)
            if oid in replaced_ops:
                if oid in new_ops_map:
                    new_ops_list.append(new_ops_map[oid])
                # else it's the ReLU, which we skip
            else:
                new_ops_list.append(op)
        
        graph.ops = new_ops_list
