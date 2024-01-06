# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

def get_extensions():
    extra_cpu_compile_args = ["-fopenmp", "-ffast-math"]

    extensions = [
        CppExtension(
            "hashing.hash_cpu",
            sources=[
                "hashing/hash_cpu.cpp"
            ],
            extra_compile_args=extra_cpu_compile_args
        ),
        CppExtension(
            "clustering.hamming.cluster_cpu",
            sources=[
               "clustering/hamming/cluster_cpu.cpp"
            ],
            extra_compile_args=extra_cpu_compile_args
        )
    ]
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extensions += [
            CUDAExtension(
                "hashing.hash_cuda",
                sources=[
                    "hashing/hash_cuda.cu",
                ],
                extra_compile_args=[]
            ),
            CUDAExtension(
                "clustering.hamming.cluster_cuda",
                sources=[
                    "clustering/hamming/cluster_cuda.cu"
                ],
                extra_compile_args=[]
            )
        ]
    return extensions

setup(
    name="ClusterUtils",
    version="1.0",
    author="ky",
    url="",
    description="PyTorch Wrapper for CUDA Functions of Clustering",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    py_modules=['clusterutils'],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
