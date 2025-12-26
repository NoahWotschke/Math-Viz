"""Setup configuration for MathVis Web UI."""

from setuptools import setup

setup(
    name="mathvis-web",
    version="0.1.0",
    description="Web UI for PDE solvers using Streamlit",
    author="Noah Wotschke",
    python_requires=">=3.8",
    install_requires=[
        "mathvis-core @ file://../mathvis-core",
        "streamlit>=1.28.0",
    ],
)
