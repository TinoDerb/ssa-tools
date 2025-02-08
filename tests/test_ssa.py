import numpy as np
import pytest
from ssa_tools.core import SSA

def test_ssa_output_classic():
    signal = np.sin(np.linspace(0, 10, 100))
    components = SSA(signal, lag=20, numComp=5, method="classic")
    assert components.shape == (100, 5)

def test_invalid_lag_classic():
    signal = np.sin(np.linspace(0, 10, 100))
    with pytest.raises(AssertionError):
        SSA(signal, lag=1, numComp=5, method="classic")

def test_ssa_output_accelerated():
    signal = np.sin(np.linspace(0, 10, 100))
    components = SSA(signal, lag=20, numComp=5, method="accelerated")
    assert components.shape == (100, 5)

def test_invalid_lag_accelerated():
    signal = np.sin(np.linspace(0, 10, 100))
    with pytest.raises(AssertionError):
        SSA(signal, lag=1, numComp=5, method="accelerated")
