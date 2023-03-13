import glob
import os
import shutil

from setuptools import setup, find_packages, Command
import sys


class PyTestCommand(Command):
    description = "run PyTest for TimeEval"
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        import pytest
        from pytest import ExitCode

        exit_code = pytest.main(
            [
                "--cov-report=term",
                "--cov-report=xml:coverage.xml",
                "--cov=tidewater",
                "tests",
            ]
        )
        if exit_code == ExitCode.TESTS_FAILED:
            raise ValueError("Tests failed!")
        elif exit_code == ExitCode.INTERRUPTED:
            raise ValueError("pytest was interrupted!")
        elif exit_code == ExitCode.INTERNAL_ERROR:
            raise ValueError("pytest internal error!")
        elif exit_code == ExitCode.USAGE_ERROR:
            raise ValueError("Pytest was not correctly used!")
        elif exit_code == ExitCode.NO_TESTS_COLLECTED:
            raise ValueError("No tests found!")
        # else: everything is fine


class MyPyCheckCommand(Command):
    description = "run MyPy for Tidewater; performs static type checking"
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from mypy.main import main as mypy

        args = ["--pretty", "tidewater", "tests"]
        mypy(None, stdout=sys.stdout, stderr=sys.stderr, args=args)


class CleanCommand(Command):
    description = "Remove build artifacts from the source tree"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        files = [".coverage*", "coverage.xml"]
        dirs = [
            "build",
            "dist",
            "*.egg-info",
            "**/__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            "**/.ipynb_checkpoints",
        ]
        for d in dirs:
            for filename in glob.glob(d):
                shutil.rmtree(filename, ignore_errors=True)

        for f in files:
            for filename in glob.glob(f):
                try:
                    os.remove(filename)
                except OSError:
                    pass


setup(
    name="tidewater",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "experiments"]),
    python_requires=">=3.8",
    test_suite="tests",
    cmdclass={"test": PyTestCommand, "typecheck": MyPyCheckCommand, "clean": CleanCommand},
)
