from setuptools import setup, find_packages

setup(
    name="ssa_tools",
    version="0.1.0",
    author="Mehieddine Derbas",
    author_email="mehyeddin.derbass@gmail.com",
    description="A Python library for Singular Spectrum Analysis (SSA)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TinoDerb/ssa_tools",  # Replace with your GitHub URL
    packages=find_packages(),
    install_requires=[
        "numpy",
        "joblib",
        "numba",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
