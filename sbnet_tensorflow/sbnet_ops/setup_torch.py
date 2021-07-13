from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    BuildExtension,
)
import os
import sysconfig
_DEBUG = False
_DEBUG_LEVEL = 0

# Common flags for both release and debug builds.
# extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
# extra_compile_args = ["-std=c++14"]
extra_compile_args = []
extra_compile_args += ["-DNDEBUG", "-O3", "-lineinfo"]
extra_compile_args = {
    'gcc': extra_compile_args,
    'nvcc': [*extra_compile_args, "--ptxas-options=-v"]
}

modules = []

if CUDA_HOME:
    modules.append(
        CUDAExtension(
            "sbnet.cuda",
            ["reduce_mask_torch.cu", "sparse_gather_torch.cu", "sparse_gather_torch_wrapper.cpp"],
			extra_compile_args=extra_compile_args,
            language='c++14'
        )
    )

setup(
    name="torchsbnet",
    packages=find_packages(where="./.."),
    package_dir={"": "./.."},
    ext_modules=modules,
    cmdclass={"build_ext": BuildExtension},
)