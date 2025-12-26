"""Setup configuration for MathVis CLI."""

from setuptools import setup

setup(
    name="mathvis-cli",
    version="0.1.0",
    description="Command-line interface for PDE solvers",
    author="Noah Wotschke",
    py_modules=["solve"],
    python_requires=">=3.8",
    install_requires=[
        "mathvis-core @ file://../mathvis-core",
    ],
    entry_points={
        "console_scripts": [
            "mathvis-solve=solve:main",
        ],
    },
)
