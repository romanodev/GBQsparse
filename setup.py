from setuptools import setup,find_packages
import os

setup(name='GBQsparse',
      version='0.9.5',
      description='Batched QR factorization of sparse matrices on GPUs. Wrapper of the CUDA library cusolverSpDcsrqrsvBatched()',
      author='Giuseppe Romano',
      author_email='romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 3.6'],
      install_requires=['pycuda',
                        'scipy',
                        'numpy',
                         ],
      license='GPLv2',\
      packages = ['GBQsparse'],
      zip_safe=False)
