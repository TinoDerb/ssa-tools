# SSA-Tools

**A comprehensive Python package for accelerated Singular Spectrum Analysis**

[![PyPI version](https://img.shields.io/pypi/v/ssa-tools.svg)](https://pypi.org/project/ssa-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SSA-Tools provides both classic and accelerated implementations of Singular Spectrum Analysis (SSA), a powerful technique for time series decomposition, trend extraction, and noise reduction. The package includes complete visualization tools, weighted correlation analysis, and automated component grouping capabilities.

## Features

- **Multiple SSA Implementations**:
  - Classic SSA with both SVD and eigenvalue decomposition
  - Accelerated SSA with multi-threading and Numba JIT compilation
  - Performance optimizations for large time series analysis

- **Comprehensive Analysis Tools**:
  - Component extraction and reconstruction
  - Weighted correlation analysis between components
  - Automated detection of related components
  - Variance analysis for components

- **Rich Visualization Suite**:
  - Reconstructed components plotting
  - Variance explained charts
  - Weighted correlation heatmaps
  - Publication-ready visualization options

- **Easy-to-Use API**:
  - Intuitive object-oriented interface
  - Sensible defaults for quick analysis
  - Advanced options for customization
  - Thorough parameter validation

## Installation

Install SSA-Tools using pip:

```bash
pip install ssa-tools
```

### Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Joblib
- Numba

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from ssa_tools import SSA

# Generate sample data (sine wave + noise)
t = np.linspace(0, 10, 1000)
signal = np.sin(t) + 0.5 * np.sin(5 * t) + 0.2 * np.random.randn(len(t))

# Initialize SSA with parameters
ssa = SSA(
    data=signal,
    lag=100,  # Window length
    numComp=8,  # Number of components to extract
    method="accelerated",  # Use accelerated implementation
    decomposition="EVD"  # Eigenvalue decomposition
)

# Perform SSA decomposition
ssa.computeSSA()

# Plot reconstructed components
ssa.plotReconstructedComponents()

# Analyze variance explained by components
ssa.plotVarianceExplained()

# Compute weighted correlations between components
ssa.computeWCA()

# Plot correlation heatmap
ssa.plotWeightedCorrelations()

# Find related components
correlated_pairs = ssa.detectCorrelatedComponents(listPairs=True)

```

## Detailed Usage

### SSA Class

The main interface to the package is the `SSA` class, which has the following key parameters:

```python
SSA(
    data,           # 1D time series as numpy array
    lag,            # Window length (2 <= lag < len(data)//2)
    numComp,        # Number of components (1 <= numComp <= lag)
    method="classic", # "classic" or "accelerated"
    decomposition="EVD", # "SVD" or "EVD"
    threshold=0.9   # Correlation threshold for grouping
)
```

### Component Analysis

After performing SSA decomposition with `computeSSA()`, you can access:

- `ssa.RC`: Reconstructed components matrix
- `ssa.eigenvalues`: Eigenvalues for each component
- `ssa.eigenvectors`: Eigenvectors from decomposition

### Weighted Correlation

Compute and analyze correlations between components:

```python
# Compute weighted correlations
ssa.computeWCA()

# Access correlation matrix
wcorr_matrix = ssa.WCorr

# Find related components
correlated_pairs = ssa.detectCorrelatedComponents()
```

### Choosing Window Length

The window length (`lag`) parameter is crucial for SSA performance:

- For periodic signals, `lag` should be larger than the period of interest
- For trend extraction, larger values of `lag` capture longer-term trends
- Typical values range from N/10 to N/2, where N is the signal length
- For initial exploration, a value around N/4 often works well

### Classic vs. Accelerated Implementation

Choose the right implementation for your needs:

- **Classic SSA**: 
  - Supports both SVD and EVD decomposition
  - Better for smaller datasets
  - More straightforward to understand and debug

- **Accelerated SSA**:
  - Only supports EVD decomposition
  - Significantly faster for large datasets
  - Uses multi-threading and JIT compilation
  - Optimized for numerical efficiency

## Advanced Examples

### Signal Filtering

```python
import numpy as np
from ssa_tools import SSA

# Generate noisy signal
t = np.linspace(0, 10, 1000)
clean = np.sin(t) + 0.5 * np.sin(5 * t)
noisy = clean + 0.5 * np.random.randn(len(t))

# SSA filtering
ssa = SSA(noisy, lag=100, numComp=2)
ssa.computeSSA()

# First two components represent filtered signal
filtered = np.sum(ssa.RC[:, :2], axis=1)

# Calculate filtering performance
mse = np.mean((filtered - clean)**2)
print(f"Mean Squared Error: {mse:.6f}")
```

### Trend Extraction

```python
import numpy as np
from ssa_tools import SSA

# Generate data with trend and seasonality
t = np.linspace(0, 10, 1000)
trend = 0.01 * t**2
seasonal = np.sin(2 * np.pi * t)
noise = 0.2 * np.random.randn(len(t))
signal = trend + seasonal + noise

# Extract trend with SSA
ssa = SSA(signal, lag=200, numComp=10)
ssa.computeSSA()

# Compute correlations
ssa.computeWCA()

# First component is usually the trend
extracted_trend = ssa.RC[:, 0]
```

More examples can be seen in examples/

## API Reference

### SSA Class

```python
from ssa_tools import SSA

# Initialize
ssa = SSA(data, lag, numComp, method, decomposition, threshold)

# Core methods
ssa.computeSSA()  # Perform decomposition
ssa.computeWCA()  # Compute weighted correlations
ssa.detectCorrelatedComponents(listPairs=False)  # Find related components

# Visualization methods
ssa.plotReconstructedComponents()  # Plot components
ssa.plotVarianceExplained()  # Plot variance distribution
ssa.plotWeightedCorrelations()  # Plot correlation heatmap
```

### Low-level functions

The package also provides direct access to the core SSA algorithms:

```python
from ssa_tools import classic_ssa, accelerated_ssa

# Classic SSA implementation
RC, eigenvectors, eigenvalues = classic_ssa(signal, lag, numComp, decomposition)

# Accelerated SSA implementation
RC, eigenvectors, eigenvalues = accelerated_ssa(signal, lag, numComp)
```

## Reference & Citation

If you use this package in a publication, please cite:

**Publication**: Accelerated Singular Spectrum Analysis and Machine Learning to investigate wood machining acoustics  
**Journal**: Mechanical Systems and Signal Processing  
**DOI**: https://doi.org/10.1016/j.ymssp.2024.111879

```bibtex
@article{SSA2024,
  title={Accelerated Singular Spectrum Analysis and Machine Learning to investigate wood machining acoustics},
  journal={Mechanical Systems and Signal Processing},
  year={2024},
  doi={10.1016/j.ymssp.2024.111879}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- SSA implementation based on methods described in ["Introducing SSA for Time Series Decomposition"](https://www.kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition)
- Accelerated implementation using Numba and multi-threading optimizations
