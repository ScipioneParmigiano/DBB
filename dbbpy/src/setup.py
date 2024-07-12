from setuptools import setup, Extension
import numpy as np
import pybind11

ext_modules = [
    Extension(
        "bindings", 
        ["src/optimizer.cpp"],  # C++ source files
        include_dirs=[
            np.get_include(),
            pybind11.get_include(),
            pybind11.get_include(user=True),
            "/home/pietro/mosek/10.2/tools/platform/linux64x86/h"  
        ],
        library_dirs=[
            "/home/pietro/mosek/10.2/tools/platform/linux64x86/bin" 
        ],
        libraries=["mosek64", "fusion64"],  
        extra_compile_args=["-std=c++11"],  
        extra_link_args=["-lmosek64", "-lfusion64"],  
        language="c++"
    ),
]

setup(
    name="bindings", 
    ext_modules=ext_modules,
    zip_safe=False,
)
