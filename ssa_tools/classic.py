import numpy as np

def classic_ssa(signal, L=10, numComp=10):
    '''
    Classic Singular Spectrum Analysis (SSA) based on:
    https://www.kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition
    
    Parameters:
        signal (array-like): The input signal.
        L (int): Window length for embedding.
        numComp (int): Number of components to reconstruct.

    Returns:
        np.ndarray: Reconstructed components as columns.
    '''
    N = len(signal)  # Lag length
    K = N - L + 1  # Embedded signal length

    # Embedding
    X = np.array([signal[i:L+i] for i in range(0, K)]).T

    # Decomposition by Singular Value Decomposition
    U, Sigma, VT = np.linalg.svd(X)

    # Reconstructed components
    RC = np.zeros((N, numComp))  
    for i in range(numComp):
        buff = Sigma[i] * np.outer(U[:, i], VT[i, :])  # Buffer matrix
        buff = buff[::-1]  # Flip for easier anti-diagonal selection
        RC[:, i] = [buff.diagonal(j).mean() for j in range(-buff.shape[0] + 1, buff.shape[1])]
    
    return RC
