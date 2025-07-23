import sys
from os.path import join
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

from Cython.Build import cythonize
import numpy


LINUX = sys.platform == "linux"
WINDOWS = sys.platform == "win32"


# Avoid a gcc warning below:
# cc1plus: warning: command line option '-Wstrict-prototypes' is valid
# for C/ObjC but not for C++
class BuildExt(build_ext):
    def build_extensions(self):
        if LINUX:
            if "-Wstrict-prototypes" in self.compiler.compiler_so:
                self.compiler.compiler_so.remove("-Wstrict-prototypes")
        super().build_extensions()


ckdtree = join("jakteristics", "ckdtree")
ckdtree_src = [
    join(ckdtree, "ckdtree", "src", x)
    for x in [
        "query.cxx",
        "build.cxx",
        "query_pairs.cxx",
        "count_neighbors.cxx",
        "query_ball_point.cxx",
        "query_ball_tree.cxx",
        "sparse_distances.cxx",
    ]
]
ckdtree_src.append(join(ckdtree, "ckdtree.pyx"))

ckdtree_includes = [
    numpy.get_include(),
    join(ckdtree, "ckdtree", "src"),
    join(ckdtree, "_lib"),
]

extra_compile_args = ["-fopenmp"]
extra_link_args = ["-fopenmp"]

if WINDOWS:
    extra_compile_args = ["/openmp"]
    extra_link_args = ["/openmp"]

ext_modules = [
    Extension(
        "jakteristics.extension",
        sources=[
            "jakteristics/extension.pyx",
            join(ckdtree, "ckdtree", "src", "query_ball_point.cxx"),
        ],
        include_dirs=ckdtree_includes,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "1")],
    ),
    Extension(
        "jakteristics.utils",
        sources=["jakteristics/utils.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "1")],
    ),
    Extension(
        "jakteristics.ckdtree.ckdtree",
        sources=ckdtree_src,
        include_dirs=ckdtree_includes,
        define_macros=[("NPY_NO_DEPRECATED_API", "1")],
    ),
]

ext_modules = cythonize(ext_modules, language_level=3)
cmdclass = {"build_ext": BuildExt}

setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)