from setuptools import find_packages, setup

# Try to import pybind11 for C++ extension
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    
    ext_modules = [
        Pybind11Extension(
            "mini_runtime",
            sources=[
                "src/bindings/bindings.cpp",
                "src/runtime/engine/engine.cpp",
                "src/runtime/memory/arena.cpp",
                "src/runtime/memory/tensors.cpp",
                "src/runtime/kernels/matmul_ref.cpp",
            ],
            include_dirs=["include"],
            cxx_std=17,
            extra_compile_args=["-O3", "-Wall", "-Wextra"],
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
