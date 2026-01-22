/**
 * @file bindings.cpp
 * @brief pybind11 bindings for the mini-runtime.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mini_runtime/engine.hpp"
#include "mini_runtime/schedule.hpp"
#include "mini_runtime/constants.hpp"
#include "mini_runtime/kernels.hpp"

namespace py = pybind11;
using namespace mini_runtime;

PYBIND11_MODULE(mini_runtime, m) {
    m.doc() = "Mini-compiler C++ runtime for executing tiled schedules";

    // Export constants
    m.attr("TILE_DIM") = TILE_DIM;
    m.attr("TILE_BYTES") = TILE_BYTES;
    m.attr("ALIGNMENT") = ALIGNMENT;

    // Engine configuration
    py::class_<Engine::Config>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("sram_bytes", &Engine::Config::sram_bytes,
                       "Size of simulated SRAM in bytes (default: 256 KiB)")
        .def_readwrite("trace", &Engine::Config::trace,
                       "Enable execution tracing (default: false)");

    // Engine statistics
    py::class_<Engine::Stats>(m, "EngineStats")
        .def_readonly("loads", &Engine::Stats::loads,
                      "Number of LOAD operations executed")
        .def_readonly("executes", &Engine::Stats::executes,
                      "Number of EXEC operations executed")
        .def_readonly("stores", &Engine::Stats::stores,
                      "Number of STORE operations executed")
        .def("__repr__", [](const Engine::Stats& s) {
            return "EngineStats(loads=" + std::to_string(s.loads) +
                   ", executes=" + std::to_string(s.executes) +
                   ", stores=" + std::to_string(s.stores) + ")";
        });

    // Schedule operation types
    py::class_<SchedLoad>(m, "SchedLoad")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t>(),
             py::arg("tensor_id"),
             py::arg("tile_row"),
             py::arg("tile_col"),
             py::arg("dst_addr"),
             py::arg("bytes"),
             py::arg("buffer") = -1)
        .def_readonly("tensor_id", &SchedLoad::tensor_id)
        .def_readonly("tile_row", &SchedLoad::tile_row)
        .def_readonly("tile_col", &SchedLoad::tile_col)
        .def_readonly("dst_addr", &SchedLoad::dst_addr)
        .def_readonly("bytes", &SchedLoad::bytes)
        .def_readonly("buffer", &SchedLoad::buffer);

    py::class_<SchedExecute>(m, "SchedExecute")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t>(),
             py::arg("m"),
             py::arg("n"),
             py::arg("k"),
             py::arg("a_addr"),
             py::arg("b_addr"),
             py::arg("acc_addr"),
             py::arg("buffer") = -1)
        .def_readonly("m", &SchedExecute::m)
        .def_readonly("n", &SchedExecute::n)
        .def_readonly("k", &SchedExecute::k)
        .def_readonly("a_addr", &SchedExecute::a_addr)
        .def_readonly("b_addr", &SchedExecute::b_addr)
        .def_readonly("acc_addr", &SchedExecute::acc_addr)
        .def_readonly("buffer", &SchedExecute::buffer);

    py::class_<SchedStore>(m, "SchedStore")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool>(),
             py::arg("tensor_id"),
             py::arg("tile_row"),
             py::arg("tile_col"),
             py::arg("src_addr"),
             py::arg("bytes"),
             py::arg("apply_relu") = false)
        .def_readonly("tensor_id", &SchedStore::tensor_id)
        .def_readonly("tile_row", &SchedStore::tile_row)
        .def_readonly("tile_col", &SchedStore::tile_col)
        .def_readonly("src_addr", &SchedStore::src_addr)
        .def_readonly("bytes", &SchedStore::bytes)
        .def_readonly("apply_relu", &SchedStore::apply_relu);

    // Main engine class
    py::class_<Engine>(m, "Engine")
        .def(py::init<const Engine::Config&>(),
             py::arg("config"),
             "Create an execution engine with the given configuration")
        
        .def("register_tensor", &Engine::register_tensor,
             py::arg("name"),
             py::arg("rows"),
             py::arg("cols"),
             "Register a tensor and return its ID")
        
        .def("set_tensor",
             [](Engine& engine, const std::string& name, py::array_t<float> arr) {
                 auto buf = arr.request();
                 if (buf.ndim != 2) {
                     throw std::invalid_argument("Expected 2D array");
                 }
                 size_t rows = buf.shape[0];
                 size_t cols = buf.shape[1];
                 
                 // Ensure contiguous C-order array
                 auto contig = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(arr);
                 if (!contig) {
                     throw std::invalid_argument("Array must be contiguous and C-order");
                 }
                 
                 engine.set_tensor(name, static_cast<const float*>(contig.data()), rows, cols);
             },
             py::arg("name"),
             py::arg("data"),
             "Set tensor data from a numpy array")
        
        .def("get_tensor",
             [](const Engine& engine, const std::string& name, 
                size_t rows, size_t cols) -> py::array_t<float> {
                 // Allocate output array
                 py::array_t<float> result({rows, cols});
                 auto buf = result.request();
                 engine.get_tensor(name, static_cast<float*>(buf.ptr), rows, cols);
                 return result;
             },
             py::arg("name"),
             py::arg("rows"),
             py::arg("cols"),
             "Get tensor data as a numpy array")
        
        .def("execute",
             [](Engine& engine, const py::list& schedule) {
                 std::vector<SchedOp> ops;
                 ops.reserve(schedule.size());
                 
                 for (const auto& item : schedule) {
                     if (py::isinstance<SchedLoad>(item)) {
                         ops.emplace_back(item.cast<SchedLoad>());
                     } else if (py::isinstance<SchedExecute>(item)) {
                         ops.emplace_back(item.cast<SchedExecute>());
                     } else if (py::isinstance<SchedStore>(item)) {
                         ops.emplace_back(item.cast<SchedStore>());
                     } else {
                         throw std::invalid_argument(
                             "Schedule contains invalid operation type");
                     }
                 }
                 
                 engine.execute(ops);
             },
             py::arg("schedule"),
             "Execute a schedule (list of SchedLoad, SchedExecute, SchedStore)")
        
        .def("stats", &Engine::stats,
             "Get execution statistics")
        
        .def("reset_stats", &Engine::reset_stats,
             "Reset execution statistics")
        
        .def_property_readonly("sram_bytes", &Engine::sram_bytes,
                               "Get SRAM size in bytes");

    // ========================================================================
    // Kernel implementation enum (for explicit benchmarking)
    // ========================================================================
    py::enum_<KernelImpl>(m, "KernelImpl")
        .value("Reference", KernelImpl::Reference, "Naive triple-nested loop")
        .value("AVX2", KernelImpl::AVX2, "AVX2+FMA optimized kernel");

    // ========================================================================
    // Kernel utility functions
    // ========================================================================
    m.def("is_avx2_available", &is_avx2_available,
          "Check if AVX2+FMA kernels are available (compile-time detection)");
    
    m.def("get_active_kernel_name", &get_active_kernel_name,
          "Get the name of the currently active kernel implementation");

    // ========================================================================
    // Direct kernel calls (for isolated benchmarking)
    // ========================================================================
    m.def("matmul_tile_bench",
          [](py::array_t<float, py::array::c_style> C,
             py::array_t<float, py::array::c_style> A,
             py::array_t<float, py::array::c_style> B,
             KernelImpl impl) {
              auto c_buf = C.request();
              auto a_buf = A.request();
              auto b_buf = B.request();
              
              if (c_buf.ndim != 2 || a_buf.ndim != 2 || b_buf.ndim != 2) {
                  throw std::invalid_argument("All arrays must be 2D");
              }
              if (c_buf.shape[0] != TILE_DIM || c_buf.shape[1] != TILE_DIM ||
                  a_buf.shape[0] != TILE_DIM || a_buf.shape[1] != TILE_DIM ||
                  b_buf.shape[0] != TILE_DIM || b_buf.shape[1] != TILE_DIM) {
                  throw std::invalid_argument("All arrays must be 32x32");
              }
              
              matmul_tile(
                  static_cast<float*>(c_buf.ptr),
                  static_cast<const float*>(a_buf.ptr),
                  static_cast<const float*>(b_buf.ptr),
                  impl
              );
          },
          py::arg("C"), py::arg("A"), py::arg("B"), py::arg("impl"),
          "Execute 32x32 matmul with explicit implementation selection");

    m.def("matmul_tile_bench",
          [](py::array_t<float, py::array::c_style> C,
             py::array_t<float, py::array::c_style> A,
             py::array_t<float, py::array::c_style> B) {
              auto c_buf = C.request();
              auto a_buf = A.request();
              auto b_buf = B.request();
              
              if (c_buf.ndim != 2 || a_buf.ndim != 2 || b_buf.ndim != 2) {
                  throw std::invalid_argument("All arrays must be 2D");
              }
              if (c_buf.shape[0] != TILE_DIM || c_buf.shape[1] != TILE_DIM ||
                  a_buf.shape[0] != TILE_DIM || a_buf.shape[1] != TILE_DIM ||
                  b_buf.shape[0] != TILE_DIM || b_buf.shape[1] != TILE_DIM) {
                  throw std::invalid_argument("All arrays must be 32x32");
              }
              
              matmul_tile(
                  static_cast<float*>(c_buf.ptr),
                  static_cast<const float*>(a_buf.ptr),
                  static_cast<const float*>(b_buf.ptr)
              );
          },
          py::arg("C"), py::arg("A"), py::arg("B"),
          "Execute 32x32 matmul with auto-dispatched implementation");

    m.def("relu_tile_bench",
          [](py::array_t<float, py::array::c_style> C, KernelImpl impl) {
              auto c_buf = C.request();
              
              if (c_buf.ndim != 2) {
                  throw std::invalid_argument("Array must be 2D");
              }
              if (c_buf.shape[0] != TILE_DIM || c_buf.shape[1] != TILE_DIM) {
                  throw std::invalid_argument("Array must be 32x32");
              }
              
              relu_tile(static_cast<float*>(c_buf.ptr), impl);
          },
          py::arg("C"), py::arg("impl"),
          "Execute 32x32 ReLU with explicit implementation selection");
}
