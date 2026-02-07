/**
 * @file threaded_engine.cpp
 * @brief Dual-threaded pipelined engine implementation.
 *
 * Key correctness mechanism: address-conflict tracking.
 *
 * The DMA thread maintains a list of operand addresses for each pushed
 * EXEC. The compute thread increments an atomic counter after finishing
 * each EXEC. Before loading into an address, the DMA thread checks that
 * no in-flight EXEC is reading from that address. If there's a conflict,
 * it spins until the compute thread has progressed past the conflicting
 * item.
 *
 * This allows true overlap when addresses don't conflict (which happens
 * with double-buffered schedules) while remaining correct for any schedule.
 */

#include "mini_runtime/threaded_engine.hpp"
#include "mini_runtime/kernels.hpp"
#include "mini_runtime/constants.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <variant>

namespace mini_runtime {

// ============================================================================
// Pre-scan: build per-EXEC metadata via reverse schedule walk
// ============================================================================

std::vector<ThreadedEngine::ExecMeta>
ThreadedEngine::prescan_schedule(const std::vector<SchedOp>& schedule) {
    std::vector<ExecMeta> meta(schedule.size());

    // Reverse scan: each STORE claims the immediately preceding EXEC group.
    std::unordered_map<uint32_t, ExecMeta> pending_stores;

    for (int i = static_cast<int>(schedule.size()) - 1; i >= 0; --i) {
        if (auto* store = std::get_if<SchedStore>(&schedule[i])) {
            ExecMeta em{};
            em.is_last_k        = true;
            em.store_tensor_id  = store->tensor_id;
            em.store_tile_row   = store->tile_row;
            em.store_tile_col   = store->tile_col;
            em.store_apply_relu = store->apply_relu;
            pending_stores[store->src_addr] = em;

        } else if (auto* exec = std::get_if<SchedExecute>(&schedule[i])) {
            auto it = pending_stores.find(exec->acc_addr);
            if (it != pending_stores.end()) {
                meta[i] = it->second;
                pending_stores.erase(it);
            }
        }
    }

    return meta;
}

// ============================================================================
// Address-conflict check
// ============================================================================

void ThreadedEngine::wait_addr_safe(uint32_t addr) const {
    while (true) {
        uint64_t done = items_completed_.load(std::memory_order_acquire);

        // Check all in-flight items (from done to items_pushed_)
        bool conflict = false;
        for (uint64_t j = done; j < items_pushed_; ++j) {
            const auto& addrs = in_flight_addrs_[j];
            if (addrs.a_addr == addr || addrs.b_addr == addr) {
                conflict = true;
                break;
            }
        }

        if (!conflict) return;

        // Spin — wait for compute thread to make progress
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#endif
    }
}

// ============================================================================
// Public API
// ============================================================================

void ThreadedEngine::execute(const std::vector<SchedOp>& schedule,
                             SRAMArena& sram,
                             TensorStorage& tensors,
                             Stats& stats) {
    // Reset state
    dma_done_.store(false, std::memory_order_relaxed);
    items_completed_.store(0, std::memory_order_relaxed);
    items_pushed_ = 0;
    in_flight_addrs_.clear();

    // Pre-reserve for in-flight tracking (count EXECs in schedule)
    size_t exec_count = 0;
    for (const auto& op : schedule) {
        if (std::holds_alternative<SchedExecute>(op)) exec_count++;
    }
    in_flight_addrs_.reserve(exec_count);

    // Pre-scan schedule for is_last_k and store metadata
    auto exec_meta = prescan_schedule(schedule);

    // Launch compute thread
    std::thread compute([this, &sram, &stats]() {
        this->compute_thread_func(sram, stats);
    });

    // Run DMA thread on current thread
    dma_thread_func(schedule, exec_meta, sram, tensors, stats);

    // Wait for compute thread to finish
    compute.join();
}

// ============================================================================
// DMA Thread
// ============================================================================

void ThreadedEngine::dma_thread_func(const std::vector<SchedOp>& schedule,
                                     const std::vector<ExecMeta>& exec_meta,
                                     SRAMArena& sram,
                                     TensorStorage& tensors,
                                     Stats& stats) {
    for (size_t i = 0; i < schedule.size(); ++i) {
        const auto& op = schedule[i];

        if (auto* load = std::get_if<SchedLoad>(&op)) {
            // ---- LOAD: wait for safe, then copy tile to SRAM ----

            // Ensure no in-flight EXEC is reading from this address
            wait_addr_safe(load->dst_addr);

            sram.validate_range(load->dst_addr, load->bytes);

            const float* src = tensors.tile_ptr(
                load->tensor_id, load->tile_row, load->tile_col);
            float* dst = sram.ptr(load->dst_addr);
            size_t src_stride = tensors.get_stride(load->tensor_id);

            for (uint32_t row = 0; row < TILE_DIM; ++row) {
                std::memcpy(dst + row * TILE_DIM,
                            src + row * src_stride,
                            TILE_DIM * sizeof(float));
            }

            stats.loads++;

        } else if (auto* exec = std::get_if<SchedExecute>(&op)) {
            // ---- EXEC: record addresses and push to compute ----
            const auto& em = exec_meta[i];

            // Track operand addresses for conflict detection
            in_flight_addrs_.push_back({exec->a_addr, exec->b_addr});

            ComputeWorkItem item{};
            item.a_addr           = exec->a_addr;
            item.b_addr           = exec->b_addr;
            item.acc_addr         = exec->acc_addr;
            item.m                = exec->m;
            item.n                = exec->n;
            item.k                = exec->k;
            item.is_first_k       = (exec->k == 0);
            item.is_last_k        = em.is_last_k;
            item.store_tensor_id  = em.store_tensor_id;
            item.store_tile_row   = em.store_tile_row;
            item.store_tile_col   = em.store_tile_col;
            item.store_apply_relu = em.store_apply_relu;

            items_pushed_++;
            compute_queue_.push(item);

        } else if ([[maybe_unused]] auto* store = std::get_if<SchedStore>(&op)) {
            // ---- STORE: wait for accumulator ready notification ----
            StoreNotification notif{};

            while (!store_queue_.try_pop(notif)) {
#if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
#endif
            }

            // Apply activation if needed
            float* src = sram.ptr(notif.acc_addr);
            if (notif.apply_relu) {
                relu_tile(src);
            }

            // Copy from SRAM to tensor storage
            float* dst_tensor = tensors.tile_ptr(
                notif.tensor_id, notif.tile_row, notif.tile_col);
            size_t dst_stride = tensors.get_stride(notif.tensor_id);

            for (uint32_t row = 0; row < TILE_DIM; ++row) {
                std::memcpy(dst_tensor + row * dst_stride,
                            src + row * TILE_DIM,
                            TILE_DIM * sizeof(float));
            }

            stats.stores++;
        }
    }

    // Signal compute thread that no more work is coming
    dma_done_.store(true, std::memory_order_release);
}

// ============================================================================
// Compute Thread
// ============================================================================

void ThreadedEngine::compute_thread_func(SRAMArena& sram, Stats& stats) {
    ComputeWorkItem item{};

    while (true) {
        if (compute_queue_.try_pop(item)) {
            // Validate SRAM ranges
            sram.validate_range(item.a_addr, TILE_BYTES);
            sram.validate_range(item.b_addr, TILE_BYTES);
            sram.validate_range(item.acc_addr, TILE_BYTES);

            const float* A = sram.ptr(item.a_addr);
            const float* B = sram.ptr(item.b_addr);
            float* C = sram.ptr(item.acc_addr);

            // Clear accumulator on first partial product
            if (item.is_first_k) {
                std::memset(C, 0, TILE_BYTES);
            }

            // Execute matmul-accumulate: C += A @ B
            matmul_tile(C, A, B);

            stats.executes++;

            // Mark this item as completed (releases operand addresses)
            items_completed_.fetch_add(1, std::memory_order_release);

            // If last K: signal DMA thread that accumulator is ready
            if (item.is_last_k) {
                StoreNotification notif{};
                notif.m          = item.m;
                notif.n          = item.n;
                notif.acc_addr   = item.acc_addr;
                notif.tensor_id  = item.store_tensor_id;
                notif.tile_row   = item.store_tile_row;
                notif.tile_col   = item.store_tile_col;
                notif.apply_relu = item.store_apply_relu;

                store_queue_.push(notif);
            }
        } else {
            // Queue empty — check if DMA is done
            if (dma_done_.load(std::memory_order_acquire)) {
                if (compute_queue_.is_empty()) {
                    break;
                }
            }
#if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
#endif
        }
    }
}

}  // namespace mini_runtime
