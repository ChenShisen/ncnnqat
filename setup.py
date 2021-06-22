from setuptools import setup, find_packages
import pathlib
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from build_helper import check_cuda_version
assert(check_cuda_version())

import os
os.system('make -j%d' % os.cpu_count())

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='ncnnqat',
    version='0.1.0',
    description='A ncnn quantization aware training tool on pytorch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ChenShisen/ncnnqat',
    author='Shisen Chen',
    author_email='napoleo54css@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords=[
        "ncnn"
        "quantization aware training",
        "deep learning",
        "neural network",
        "CNN",
        "machine learning",
    ],
    packages=find_packages(),

    python_requires='>=3.5, <4',
    install_requires=[
      "torch >= 1.5",
      "numpy >= 1.18.1",
      "onnx >= 1.7.0",
      "onnx-simplifier >= 0.3.5"
    ],
    extras_require={
        'test': ["torchvision>=0.4",
                 "nose",
                 "ddt"
                 ],
        'docs': [
            'sphinx==2.4.4',
            'sphinx_rtd_theme'
        ]
    },
    ext_modules=[
        CUDAExtension(
            #name="quant_impl",
            name="quant_cuda",
            sources=[
                "./src/fake_quantize.cpp",
            ],
            libraries=['quant_cuda'],
            library_dirs=['obj'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    #test_suite="ncnnqat.test.test_cifar10",
)
