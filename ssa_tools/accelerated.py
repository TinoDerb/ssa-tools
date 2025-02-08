import numpy as np
from joblib import Parallel, delayed, cpu_count
import numba as nb
import threading

# JIT-accelerated functions
@nb.jit(nb.float64[:, :](nb.float64[:, ::1], nb.float64[:, ::1]), fastmath=True, nopython=True)
def fastDot(X, Y):
    return np.dot(X, Y)

@nb.jit(nb.float64[:,:](nb.float64[:], nb.float64[:]), nopython=True)
def fastBuf(principalComponentColumn, eigenVectorsRow):
    return np.outer(principalComponentColumn, eigenVectorsRow)[::-1]

@nb.jit(nb.float64[:,:](nb.float64[:,:]), nopython=True)
def fastCovariance(X):
    return np.cov(X)

@nb.jit(nopython=True)
def fastMean(X):
    return X.mean()

# Helper functions for multi-threading
def getColumn(s, K, m):
    return s[m:K+m]

def laggingFunction(idx, LaggedMatrix, s, K):
    LaggedMatrix[:, idx] = getColumn(s, K, idx)

def calcRC(antiDiagonal):
    return fastMean(antiDiagonal)

def accelerated_ssa(signal, L=10, numComp=10):
    '''
    Accelerated Singular Spectrum Analysis (SSA) using Numba & Joblib.

    Parameters:
        signal (array-like): The input signal.
        L (int): Window length for embedding.
        numComp (int): Number of components to reconstruct.

    Returns:
        np.ndarray: Reconstructed components as columns.
    '''
    N = len(signal)
    K = N - L + 1

    # Embedding using multi-threading
    X = np.zeros((K, L))
    threads = []
    for m in range(L):
        threads.append(threading.Thread(target=laggingFunction, args=(m, X, signal, K)))
        threads[m].start()
    for m in range(L):
        threads[m].join()

    # Eigenvalue decomposition
    eigenValues, eigenVectors = np.linalg.eigh(fastCovariance(X.T) / K)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues, eigenVectors = eigenValues[idx], eigenVectors[:, idx]

    # Convert to C-contiguous arrays for speed
    eigenVectors, X = np.ascontiguousarray(eigenVectors), np.ascontiguousarray(X)

    # Compute principal components
    PC = fastDot(X, eigenVectors)

    # Reconstructed components
    RC = np.zeros((N, numComp), dtype=np.float64)
    for i in range(numComp):
        buf = fastBuf(PC[:, i], eigenVectors[:, i].T)
        Diag0, Diag1 = -buf.shape[0] + 1, buf.shape[1]
        RC[:, i] = Parallel(n_jobs=cpu_count())(delayed(calcRC)(buf.diagonal(j)) for j in range(Diag0, Diag1))

    return RC
