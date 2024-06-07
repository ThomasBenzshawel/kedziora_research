from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from glob import glob

__version__ = '0.0.3'

ext_modules = [
    Pybind11Extension(  
        'manifold',
        glob('src/*.cpp'),
        define_macros=[('VERSION_INFO', __version__)],
        extra_compile_args=['-g']
    ),
]

setup(
    name='manifold',
    version=__version__,
    author='Aidan Schneider',
    description='Processes STL faces and vertices to evaluate its printability',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.7',
)