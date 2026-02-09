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
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>
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

    /// Nanosecond timestamp type
    using TimePoint = std::chrono::steady_clock::time_point;

    /// A [start, end) time interval in nanoseconds relative to epoch
    struct Interval {
        uint64_t start_ns;
        uint64_t end_ns;
    };

    /**
     * @struct Stats
     * @brief Execution statistics for the threaded engine.
     */
    struct Stats {
        uint64_t loads = 0;
        uint64_t executes = 0;
        uint64_t stores = 0;

        // --- Overlap metrics (populated after execute completes) ---
        uint64_t dma_busy_ns = 0;      ///< Total ns DMA thread spent on LOAD/STORE
        uint64_t compute_busy_ns = 0;  ///< Total ns Compute thread spent on EXEC
        uint64_t overlap_ns = 0;       ///< Total ns both threads were busy simultaneously
        uint64_t total_ns = 0;         ///< Wall-clock ns for the entire execution

        /// Fraction of time both threads were active (0.0 - 1.0)
        double overlap_ratio() const {
            return total_ns > 0 ? static_cast<double>(overlap_ns) / total_ns : 0.0;
        }
        /// Fraction of time DMA thread was busy
        double dma_utilization() const {
            return total_ns > 0 ? static_cast<double>(dma_busy_ns) / total_ns : 0.0;
        }
        /// Fraction of time Compute thread was busy
        double compute_utilization() const {
            return total_ns > 0 ? static_cast<double>(compute_busy_ns) / total_ns : 0.0;
        }
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

    /// Time origin for this execution (set at start of execute())
    TimePoint epoch_;

    /// Recorded intervals (thread-local during execution, read after join)
    std::vector<Interval> dma_intervals_;
    std::vector<Interval> compute_intervals_;

    /// Get nanoseconds since epoch_
    uint64_t now_ns() const {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - epoch_).count());
    }

    /**
     * @brief Compute total overlap between two sorted interval lists.
     *
     * Both lists must be sorted by start_ns (which they naturally are
     * since each thread records intervals sequentially).
     *
     * @return Total nanoseconds of overlap.
     */
    static uint64_t compute_overlap(const std::vector<Interval>& a,
                                    const std::vector<Interval>& b);
};

}  // namespace mini_runtime
