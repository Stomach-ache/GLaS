#from setuptools import setup
#from Cython.Build import cythonize
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


ext_module = Extension(
    "cmetrics",
    ["cmetrics.pyx"],
    language="c++",
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"]
)

setup(
    ext_modules = cythonize(ext_module),
    include_dirs=[numpy.get_include()]
)
