from distutils.core import setup
from distutils.extension import Extension
import os.path

import numpy

setup(
    name = "pyFM",
    ext_modules = [Extension("pylibfm/pyfm_fast", ["pylibfm/pyfm_fast.c"],
                             include_dirs=[numpy.get_include()],
                             extra_link_args=["-O3","-ffast-math"]
                             ),
                  ],
    version = '0.2.1',
    description="A python implementation of libFM",
    author='Corey Lynch',
    author_email='coreylynch9@gmail.com',
    url="https://github.com/pprett/pyFM/blob",
    packages=['pylibfm'],
)
