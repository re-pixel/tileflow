import platform
import subprocess
import sys

from setuptools import find_packages, setup


def check_avx2_support():
    """Check if the compiler supports AVX2 and FMA flags."""
    if platform.system() == "Windows":
        # On Windows, assume modern MSVC supports AVX2
        return True
    
    try:
        # Try to compile a simple AVX2 test
        result = subprocess.run(
            ["gcc", "-mavx2", "-mfma", "-x", "c", "-c", "-", "-o", "/dev/null"],
            input=b"int main() { return 0; }",
            capture_output=True,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


# Try to import pybind11 for C++ extension
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    
    # Base source files (always included)
    sources = [
        "src/bindings/bindings.cpp",
        "src/runtime/engine/engine.cpp",
        "src/runtime/memory/arena.cpp",
        "src/runtime/memory/tensors.cpp",
        "src/runtime/kernels/matmul_ref.cpp",
        "src/runtime/kernels/dispatcher.cpp",
    ]
    
    # Compiler flags
    extra_compile_args = ["-O3", "-Wall", "-Wextra"]
    
    # Check for AVX2 support and add optimized kernel
    if check_avx2_support():
        print("AVX2+FMA support detected - enabling optimized kernels", file=sys.stderr)
        sources.append("src/runtime/kernels/matmul_avx2.cpp")
        extra_compile_args.extend(["-mavx2", "-mfma"])
    else:
        print("AVX2+FMA not available - using reference kernels only", file=sys.stderr)
    
    ext_modules = [
        Pybind11Extension(
            "mini_runtime",
            sources=sources,
            include_dirs=["include"],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
        ),
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:
    # pybind11 not installed, skip C++ extension
    ext_modules = []
    cmdclass = {}


setup(
    name="mini-compiler",
    version="0.1.0",
    description="Mini compiler: graph IR -> tiled uops -> simulated runtime",
    author="Relja",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "dev": [
            "pytest>=7",
            "pybind11>=2.11",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
