from setuptools import setup
import os
from setuptools import dist

dist.Distribution().fetch_build_eggs(['Cython', 'numpy'])

import numpy
from Cython.Build import cythonize

required = [
    "pytest", "cython", "numpy<=1.19", "tensorflow==1.14", "numba==0.53.1",
    "sympy", "pandas", "scikit-learn", "click", "deap", "pathos", "seaborn",
    "progress", "tqdm", "commentjson", "PyYAML", "prettytable", "gast==0.2.2",
    "psutil"
]

extras = {
    "control": [
        "mpi4py", "gym[box2d]==0.15.4", "pybullet",
        "stable-baselines[mpi]==2.10.0"
    ],
    "regression": []
}
extras['all'] = list(set([item for group in extras.values()
                          for item in group]))

setup(name='ssde',
      version='1.0dev',
      description='Deep symbolic solver for differential equations.',
      author='Anonymous"',
      packages=['ssde'],
      setup_requires=["numpy", "Cython"],
      ext_modules=cythonize([
          os.path.join('ssde', 'cyfunc.pyx'),
          os.path.join('ssde', 'cyfunc_recursion.pyx')
      ],
                            annotate=True),
      include_dirs=[numpy.get_include()],
      install_requires=required,
      extras_require=extras)
