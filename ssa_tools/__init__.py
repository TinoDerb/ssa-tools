"""
SSA-Tools: A comprehensive Python package for accelerated Singular Spectrum Analysis
=================================================================================

This package provides tools for Singular Spectrum Analysis (SSA), a technique
for time series decomposition, trend extraction, and noise reduction.

Main Features
------------
- Classic and accelerated SSA implementations
- Comprehensive analysis tools
- Rich visualization suite
- Easy-to-use API

Available modules:
-----------------
- classic: Classic SSA implementation
- accelerated: Accelerated SSA with Numba and multi-threading
- core: Main SSA class with analysis and visualization tools
"""

__version__ = '0.1.0'

# Import main components to make them available at package level
from .classic import classic_ssa
from .accelerated import accelerated_ssa
from .core import SSA

__all__ = ['classic_ssa', 'accelerated_ssa', 'SSA']