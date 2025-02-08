import numpy as np
from .classic import classic_ssa
from .accelerated import accelerated_ssa

def SSA(signal, lag, numComp, method="classic"):
    """
    Wrapper function to perform Singular Spectrum Analysis (SSA).
    
    Parameters:
        signal (numpy.ndarray): The input time-series signal.
        lag (int): Window length for embedding.
        numComp (int): Number of components to reconstruct.
        method (str): Either 'classic' or 'accelerated' SSA.

    Returns:
        np.ndarray: Reconstructed components as columns.
    """
    assert isinstance(signal, np.ndarray) and signal.ndim == 1, "signal must be 1D and of type ndarray"
    assert isinstance(lag, int) and 2 <= lag < len(signal) // 2, "Lag must be between 2 and len(signal)//2"
    assert isinstance(numComp, int) and 1 <= numComp <= lag, "Number of components must be between 1 and lag"
    
    if method == "classic":
        return classic_ssa(signal, lag, numComp)
    elif method == "accelerated":
        return accelerated_ssa(signal, lag, numComp)
    else:
        raise ValueError("Options are either 'classic' or 'accelerated'")
