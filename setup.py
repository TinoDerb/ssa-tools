from setuptools import setup, find_packages

# Read in the README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ssa-tools",
    version="1.0.0",
    author="Mehieddine Derbas",
    author_email="mehyeddin.derbass@gmail.com",
    description="A comprehensive Python package for Singular Spectrum Analysis with accelerated implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TinoDerb/ssa-tools",
    project_urls={
        "Bug Tracker": "https://https://github.com/TinoDerb/ssa_tools/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "joblib>=1.0.0",
        "numba>=0.53.0",
    ],
    keywords="singular spectrum analysis, signal processing, time series, decomposition, trend extraction, noise reduction")
