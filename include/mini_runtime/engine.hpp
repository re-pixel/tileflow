/**
 * @file engine.hpp
 * @brief Schedule execution engine.
 *
 * The Engine consumes a schedule (list of SchedOps) and executes them
 * sequentially. It manages SRAM and tensor storage, dispatching each
 * operation to the appropriate handler.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mini_runtime/arena.hpp"
#include "mini_runtime/schedule.hpp"
#include "mini_runtime/tensors.hpp"

namespace mini_runtime {

/**
 * @class Engine
 * @brief Executes compiled schedules against simulated memory hierarchy.
 */
class Engine {
public:
    /**
     * @struct Config
     * @brief Engine configuration parameters.
     */
    struct Config {
        size_t sram_bytes = 256 * 1024;  ///< SRAM size (default 256 KiB)
        bool   trace = false;             ///< Enable execution tracing
    };

    /**
     * @struct Stats
     * @brief Execution statistics.
     */
    struct Stats {
        uint64_t loads = 0;      ///< Number of LOAD operations executed
        uint64_t executes = 0;   ///< Number of EXEC operations executed
        uint64_t stores = 0;     ///< Number of STORE operations executed
    };

    /**
     * @brief Construct an engine with the given configuration.
     * @param config Engine configuration.
     */
    explicit Engine(const Config& config);

    // Non-copyable, movable
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) noexcept = default;
    Engine& operator=(Engine&&) noexcept = default;

    /**
     * @brief Register a tensor for use in execution.
     * @param name Unique tensor name (must match schedule references).
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return Tensor ID.
     */
    uint32_t register_tensor(const std::string& name, size_t rows, size_t cols);

    /**
     * @brief Set tensor data from external source.
     * @param name Tensor name.
     * @param data Source data (row-major).
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    void set_tensor(const std::string& name, const float* data, size_t rows, size_t cols);

    /**
     * @brief Get tensor data to external buffer.
     * @param name Tensor name.
     * @param data Destination buffer.
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    void get_tensor(const std::string& name, float* data, size_t rows, size_t cols) const;

    /**
     * @brief Execute a schedule.
     *
     * Clears SRAM, then dispatches each operation in sequence.
     *
     * @param schedule List of scheduled operations.
     */
    void execute(const std::vector<SchedOp>& schedule);

    /**
     * @brief Get execution statistics.
     * @return Stats struct with operation counts.
     */
    Stats stats() const noexcept { return stats_; }

    /**
     * @brief Reset execution statistics.
     */
    void reset_stats() noexcept { stats_ = Stats{}; }

    /**
     * @brief Get the SRAM size.
     */
    size_t sram_bytes() const noexcept { return sram_.total_bytes(); }

private:
    Config config_;
    SRAMArena sram_;
    TensorStorage tensors_;
    Stats stats_;

    void dispatch(const SchedLoad& op);
    void dispatch(const SchedExecute& op);
    void dispatch(const SchedStore& op);
};

}  // namespace mini_runtime
