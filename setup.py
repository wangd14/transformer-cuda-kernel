from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='softmax',
    ext_modules=[
        CUDAExtension('softmax', [
            'softmax_extension.cpp',
            'softmax_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })