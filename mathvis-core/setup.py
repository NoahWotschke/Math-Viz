"""Setup configuration for mathvis-core solver library."""

from setuptools import setup, find_packages

setup(
    name="mathvis-core",
    version="0.1.0",
    description="PDE solver library for numerical solutions on various domains",
    author="Noah Wotschke",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.60.0",
    ],
)
