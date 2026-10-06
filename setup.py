"""Build script for the compiled pybind11 extension modules.

All package metadata lives in pyproject.toml; this file only defines the
C++ extensions, which cannot be expressed declaratively.

Optimisation and OpenMP flags are chosen at build time based on the compiler
actually in use: each candidate flag is test-compiled against the real
compiler and dropped if it is rejected, so the same script builds with GCC,
Clang (including Apple Clang with Homebrew libomp), Intel (icc/icx) and MSVC.
"""

import os
import shlex
import subprocess
import sys
import tempfile

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError

# The #error makes this a functional test: it fails unless the flag really
# enables OpenMP, guarding against compilers that only warn on unknown flags.
OPENMP_TEST_SOURCE = """\
#include <omp.h>
#ifndef _OPENMP
#error OpenMP support is not enabled
#endif
int main() { return omp_get_max_threads() > 0 ? 0 : 1; }
"""


def try_compile(compiler, flags, source="int main() { return 0; }\n"):
    """Return True if a trivial file compiles with the given extra flags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "flag_check.cpp")
        with open(src, "w", encoding="utf-8") as f:
            f.write(source)
        try:
            compiler.compile([src], output_dir=tmpdir, extra_postargs=list(flags))
        except CompileError:
            return False
        return True


def homebrew_libomp_prefix():
    """Locate Homebrew's libomp, which Apple Clang needs for OpenMP."""
    try:
        prefix = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
        if os.path.isdir(prefix):
            return prefix
    except (OSError, subprocess.CalledProcessError):
        pass
    for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        if os.path.isdir(prefix):
            return prefix
    return None


class BuildExt(build_ext):
    """Apply per-compiler flags to every extension before building."""

    # Tried individually on GCC-style compilers; unsupported flags are dropped.
    candidate_opt_flags = [
        "-O2",
        "-march=native",
        "-mtune=native",
        "-funroll-loops",
        "-fcf-protection=none",
        "-fno-stack-protector",
    ]

    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            compile_args, link_args = self.msvc_flags()
        else:
            compile_args, link_args = self.unix_flags()
        for ext in self.extensions:
            ext.extra_compile_args = compile_args
            ext.extra_link_args = link_args
        super().build_extensions()

    def msvc_flags(self):
        compile_args = ["/O2"]
        # /openmp:llvm (VS 2019 16.9+) supports unsigned loop counters; plain
        # /openmp is the OpenMP 2.0 fallback. MSVC links its OpenMP runtime
        # automatically, so no link flags are needed.
        for omp_flag in ("/openmp:llvm", "/openmp"):
            if try_compile(self.compiler, [omp_flag], OPENMP_TEST_SOURCE):
                compile_args.append(omp_flag)
                return compile_args, []
        raise RuntimeError("This MSVC installation does not support OpenMP (/openmp).")

    def unix_flags(self):
        compile_args = [f for f in self.candidate_opt_flags if try_compile(self.compiler, [f])]
        omp_compile, omp_link = self.unix_openmp_flags()
        # GCC OpenMP offload can crash with some LTO configurations; keep LTO
        # disabled by default but overridable via the environment.
        lto_flags = [
            f
            for f in shlex.split(os.environ.get("NOISYCIRCUITS_LTO_FLAGS", "-fno-lto"))
            if try_compile(self.compiler, [f])
        ]
        return compile_args + lto_flags + omp_compile, omp_link + lto_flags

    def unix_openmp_flags(self):
        # GCC, Clang and Intel icx take -fopenmp; classic Intel icc prefers
        # -qopenmp. Apple Clang recognises neither and needs the preprocessor
        # form plus Homebrew's libomp for the header and library.
        for flag in ("-fopenmp", "-qopenmp"):
            if try_compile(self.compiler, [flag], OPENMP_TEST_SOURCE):
                return [flag], [flag]
        if sys.platform == "darwin":
            prefix = homebrew_libomp_prefix()
            if prefix is not None:
                omp_compile = [
                    "-Xpreprocessor",
                    "-fopenmp",
                    "-I" + os.path.join(prefix, "include"),
                ]
                if try_compile(self.compiler, omp_compile, OPENMP_TEST_SOURCE):
                    return omp_compile, ["-L" + os.path.join(prefix, "lib"), "-lomp"]
        raise RuntimeError(
            "No working OpenMP flag was found for this compiler. On macOS, "
            "install libomp (brew install libomp) or build with GCC."
        )


ext_modules = [
    Extension(
        "simulator",
        ["./src/NoisyCircuits/utils/custom/src/Simulator.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
    Extension(
        "simulator_mpi",
        [
            "./src/NoisyCircuits/utils/custom/src/SimulatorMPI.cpp",
            "./src/NoisyCircuits/utils/custom/src/Communicator.cpp"
        ],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
    Extension(
        "measurement_error_applicator",
        ["./src/NoisyCircuits/utils/custom/src/MeasurementErrorApplicator.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExt})
