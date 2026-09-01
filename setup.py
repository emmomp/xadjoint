"""Setuptools configuration for the xadjoint package."""

from pathlib import Path

from setuptools import setup


HERE = Path(__file__).parent


setup(
    name="xadjoint",
    version="0.1.0",
    description="Read and analyse MITgcm adjoint sensitivities",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Emma Boland",
    author_email="emmomp@bas.ac.uk",
    url="https://github.com/emmomp/xadjoint",
    packages=["xadjoint"],
    package_dir={"xadjoint": "."},
    package_data={"xadjoint": ["inputs_*.json"]},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "cartopy",
        "ecco_v4_py",
        "matplotlib",
        "numpy",
        "xarray",
        "xmitgcm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
)
