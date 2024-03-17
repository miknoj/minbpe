from setuptools import setup
from Cython.Build import cythonize


cython_module_list = [
    "minbpe/base.py",
    "minbpe/basic.py",
    "train.py",
]

setup(
    name="minbpe",
    ext_modules=cythonize(module_list=cython_module_list, language="c++"),
)
