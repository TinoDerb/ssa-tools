import numpy as np
import pytest
from ssa_tools.core import SSA

# Sample signal: Sine + Cosine + Noise
np.random.seed(42)
signal = np.sin(np.linspace(0, 2 * np.pi, 100)) + np.cos(np.linspace(0, 10 * np.pi, 100))+ 0.1 * np.random.randn(100)

@pytest.mark.parametrize("method", ["classic", "accelerated"])
def test_valid_initialization(method):
    ssa = SSA(data=signal, lag=20, numComp=5, method=method)
    assert ssa.L == 20
    assert ssa.numComp == 5
    assert ssa.method == method
    assert ssa.K == len(signal) - 20 + 1

def test_invalid_data_type():
    with pytest.raises(AssertionError):
        SSA(data="not_array", lag=10, numComp=2)

def test_invalid_lag():
    with pytest.raises(AssertionError):
        SSA(data=signal, lag=1, numComp=2)

def test_invalid_num_components():
    with pytest.raises(AssertionError):
        SSA(data=signal, lag=20, numComp=30)

def test_invalid_method():
    with pytest.raises(AssertionError):
        SSA(data=signal, lag=20, numComp=5, method="fastest")

def test_invalid_decomposition():
    with pytest.raises(AssertionError):
        SSA(data=signal, lag=20, numComp=5, decomposition="XYZ")

def test_accelerated_ssa_raises_with_svd():
    with pytest.raises(AssertionError):
        ssa = SSA(data=signal, lag=20, numComp=5, method="accelerated", decomposition="SVD")
        ssa.computeSSA()

@pytest.mark.parametrize("method", ["classic", "accelerated"])
def test_ssa_computation_shapes(method):
    ssa = SSA(data=signal, lag=20, numComp=5, method=method)
    ssa.computeSSA()
    assert ssa.RC.shape == (len(signal), 5)
    assert ssa.eigenvectors.shape[1] >= 5
    assert len(ssa.eigenvalues) >= 5

def test_weighted_correlation_matrix():
    ssa = SSA(data=signal, lag=20, numComp=5)
    ssa.computeSSA()
    ssa.computeWCA()
    assert ssa.WCorr.shape == (5, 5)
    assert np.allclose(np.diag(ssa.WCorr), 1)
    assert np.all((ssa.WCorr >= 0) & (ssa.WCorr <= 1))

def test_detect_correlated_components():
    ssa = SSA(data=signal, lag=20, numComp=5)
    ssa.computeSSA()
    ssa.computeWCA()
    pairs = ssa.detectCorrelatedComponents()
    assert isinstance(pairs, list) or pairs is None

def test_plot_functions_run():
    ssa = SSA(data=signal, lag=20, numComp=5)
    ssa.computeSSA()
    ssa.computeWCA()
    ssa.plotReconstructedComponents()
    ssa.plotVarianceExplained()
    ssa.plotWeightedCorrelations()
    
def all_equivalent():
    myEVD_SSA = SSA(data=signal, lag=20, numComp=5, method = "classic", decomposition = "SVD")
    myEVD_SSA.computeSSA()
    myEVD_SSA.computeWCA()
    myEVD_ACCSSA = SSA(data=signal, lag=20, numComp=5, method = "accelerated", decomposition = "SVD")
    myEVD_ACCSSA.computeSSA()
    myEVD_ACCSSA.computeWCA()
    mySVD_SSA = SSA(data=signal, lag=20, numComp=5, method = "accelerated", decomposition = "SVD")
    mySVD_SSA.computeSSA()
    mySVD_SSA.computeWCA()
    # Compare myEVD_SSA and myEVD_ACCSSA
    assert np.allclose(np.abs(myEVD_SSA.RC), np.abs(myEVD_ACCSSA.RC)), "RC mismatch"
    assert np.allclose(np.abs(myEVD_SSA.WCorr), np.abs(myEVD_ACCSSA.WCorr)), "WCorr mismatch"
    assert np.allclose(myEVD_SSA.eigenvalues, myEVD_ACCSSA.eigenvalues), "Eigenvalues mismatch"
    assert np.allclose(np.abs(myEVD_SSA.eigenvectors), np.abs(myEVD_ACCSSA.eigenvectors)), "Eigenvectors mismatch"
    # Compare myEVD_SSA and mySVD_SSA
    assert np.allclose(np.abs(myEVD_SSA.RC), np.abs(mySVD_SSA.RC)), "RC mismatch"
    assert np.allclose(np.abs(myEVD_SSA.WCorr), np.abs(mySVD_SSA.WCorr)), "WCorr mismatch"
    assert np.allclose(myEVD_SSA.eigenvalues, mySVD_SSA.eigenvalues), "Eigenvalues mismatch"
    assert np.allclose(np.abs(myEVD_SSA.eigenvectors), np.abs(mySVD_SSA.eigenvectors)), "Eigenvectors mismatch"
    