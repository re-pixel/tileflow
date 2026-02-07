/**
 * @file threaded_engine.hpp
 * @brief Dual-threaded pipelined execution engine.
 *
 * Separates data movement (DMA) from compute into two threads,
 * mirroring real accelerator designs (TPUs, Tenstorrent Tensix).
 *
 * Architecture:
 *   DMA Thread:     processes SchedLoad and SchedStore operations
 *   Compute Thread: processes SchedExecute operations (matmul kernels)
 *
 * Communication:
 *   compute_queue_: DMA -> Compute (work items to execute)
 *   store_queue_:   Compute -> DMA (accumulators ready to store)
 *   items_completed_: Compute -> DMA (how many EXECs have finished)
 *
 * Safety:
 *   Before each LOAD, the DMA thread checks whether the destination
 *   address is still being read by an in-flight EXEC. If so, it spins
 *   until the compute thread finishes that item. This allows overlap
 *   when addresses don't conflict (double-buffered schedules) while
 *   remaining correct for single-buffered schedules.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "mini_runtime/arena.hpp"
#include "mini_runtime/ring_buffer.hpp"
#include "mini_runtime/schedule.hpp"
#include "mini_runtime/tensors.hpp"
#include "mini_runtime/work_item.hpp"

namespace mini_runtime {

/**
 * @class ThreadedEngine
 * @brief Pipelined DMA + Compute engine for overlapped execution.
 */
class ThreadedEngine {
public:
    /// Ring buffer depth for inter-thread queues (power of 2)
    static constexpr size_t QUEUE_DEPTH = 16;

    /**
     * @struct Stats
     * @brief Execution statistics for the threaded engine.
     */
    struct Stats {
        uint64_t loads = 0;
        uint64_t executes = 0;
        uint64_t stores = 0;
    };

    /**
     * @struct ExecMeta
     * @brief Pre-computed metadata for each SchedExecute in the schedule.
     */
    struct ExecMeta {
        bool     is_last_k = false;
        uint32_t store_tensor_id = 0;
        uint32_t store_tile_row = 0;
        uint32_t store_tile_col = 0;
        bool     store_apply_relu = false;
    };

    /**
     * @struct InFlightAddrs
     * @brief Operand addresses of a pushed (possibly in-flight) EXEC.
     */
    struct InFlightAddrs {
        uint32_t a_addr;
        uint32_t b_addr;
    };

    void execute(const std::vector<SchedOp>& schedule,
                 SRAMArena& sram,
                 TensorStorage& tensors,
                 Stats& stats);

private:
    static std::vector<ExecMeta> prescan_schedule(
        const std::vector<SchedOp>& schedule);

    void dma_thread_func(const std::vector<SchedOp>& schedule,
                         const std::vector<ExecMeta>& exec_meta,
                         SRAMArena& sram,
                         TensorStorage& tensors,
                         Stats& stats);

    void compute_thread_func(SRAMArena& sram, Stats& stats);

    /**
     * @brief Wait until no in-flight EXEC is reading from `addr`.
     *
     * Called by DMA thread before each LOAD to prevent overwriting
     * data that the compute thread is still using.
     */
    void wait_addr_safe(uint32_t addr) const;

    /// DMA -> Compute queue
    RingBuffer<ComputeWorkItem, QUEUE_DEPTH> compute_queue_;

    /// Compute -> DMA queue (store readiness notifications)
    RingBuffer<StoreNotification, QUEUE_DEPTH> store_queue_;

    /// Signals that the DMA thread has finished pushing all work
    std::atomic<bool> dma_done_{false};

    /// Number of EXEC items the compute thread has completed.
    /// Written by compute thread, read by DMA thread.
    std::atomic<uint64_t> items_completed_{0};

    /// Operand addresses of all pushed EXECs, indexed by push order.
    /// Only accessed by DMA thread for writes; read with items_completed_
    /// to determine which items are still in-flight.
    std::vector<InFlightAddrs> in_flight_addrs_;

    /// Number of EXECs pushed so far (DMA thread only).
    uint64_t items_pushed_{0};
};

}  // namespace mini_runtime
