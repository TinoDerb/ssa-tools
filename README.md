SSA Tools: Accelerated Singular Spectrum Analysis

A Python package for Singular Spectrum Analysis (SSA) with accelerated performance for large datasets.
Overview

Singular Spectrum Analysis (SSA) is a powerful method for time-series decomposition based on spectral decomposition. However, the classical implementation of SSA is computationally expensive, limiting its application to small datasets.

This package provides an accelerated SSA function, allowing SSA to be applied to large datasets within a reasonable time.

🔹 Why Use This Package?

✔ Fast & efficient implementation

✔ Works with large time-series data, even with sensor signals with sampling up to 2 MHz!

✔ Supports both classic and accelerated SSA

Installation

You can install the package using:

```
pip install git+https://github.com/TinoDerb/ssa_tools.git
```

Or, if using PyPI:

```
pip install ssa_tools
```

# Usage

```
import numpy as np
import ssa_tools
signal = np.sin(np.linspace(0, 10, 100)) # Generate a sample signal
components = ssa_tools.SSA(signal, lag=20, numComp=5, method="classic") # Apply SSA (classic method)
components_fast = ssa_tools.SSA(signal, lag=20, numComp=5, method="accelerated") # Apply SSA (accelerated method)
print(components.shape)  # (100, 5)
print(components_fast.shape)  # (100, 5)
```

# Features

    Classic SSA: Based on Singular Value Decomposition (SVD)
    Accelerated SSA: Uses Numba, Joblib, and parallel processing
    Scalable: Handles large time-series efficiently
    Easy-to-use API

# Reference & Citation

If you use this package in a publication, please cite:

Publication: 
Journal: Mechanical Systems and Signal Processing
DOI: https://doi.org/10.1016/j.ymssp.2024.111879

# License

This package is licensed under the MIT License.

# Future version

In future versions, I plan to extend the capabilities to include:

✅ Variance explained by each component

✅ Weighted Correlation Analysis between the reconstructed components and the original data

✅ Automatic parameter selection for SSA optimization
