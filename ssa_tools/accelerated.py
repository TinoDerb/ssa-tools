import numpy as np
from joblib import Parallel, delayed, cpu_count
import numba as nb
import threading

# JIT-accelerated functions
@nb.jit(nb.float64[:, :](nb.float64[:, ::1], nb.float64[:, ::1]), fastmath=True, nopython=True)
def fastDot(X, Y):
    return np.dot(X, Y)                          # Optimized matrix multiplication with Numba

@nb.jit(nb.float64[:,:](nb.float64[:], nb.float64[:]), nopython=True)
def fastBuf(principalComponentColumn, eigenVectorsRow):
    return np.outer(principalComponentColumn, eigenVectorsRow)[::-1]  # Optimized outer product with flipping

@nb.jit(nb.float64[:,:](nb.float64[:,:]), nopython=True)
def fastCovariance(X):
    return np.cov(X)                             # Optimized covariance calculation

@nb.jit(nopython=True)
def fastMean(X):
    return X.mean()                              # Optimized mean calculation

def calcRC(antiDiagonal):
    return fastMean(antiDiagonal)                # Calculate mean for diagonal averaging
    
# Helper functions for multi-threading
def getColumn(s, L, m):
    return s[m:m + L]                            # Extract window of length L starting at position m

def laggingFunction(idx, LaggedMatrix, s, L):
    LaggedMatrix[:, idx] = getColumn(s, L, idx)  # Fill column idx of matrix with lagged values

def accelerated_ssa(signal, lag=10, numComp=10):
    '''
    Accelerated Singular Spectrum Analysis (SSA) using Numba & Joblib.
    
    This optimized implementation leverages:
    1. Numba JIT compilation for numerical operations
    2. Multi-threading for trajectory matrix construction
    3. Parallel processing for diagonal averaging
    4. Memory layout optimization for contiguous arrays
    
    Parameters:
        signal (array-like): The input time series signal to decompose
        lag (int, optional): Window length for embedding. Defaults to 10.
                          (Recommended: 2 <= lag <= N/2)
        numComp (int, optional): Number of components to reconstruct. Defaults to 10.
                                (Must be <= lag)
    
    Returns:
        tuple:
            - RC (np.ndarray): Reconstructed components as columns, shape (N, numComp)
            - eigenvectors (np.ndarray): The eigenvectors from decomposition
            - eigenvalues (np.ndarray): The eigenvalues from decomposition
    
    Notes:
        This function performs the same SSA algorithm as classic_ssa but with
        significant performance optimizations through parallel processing and
        JIT compilation.
    '''
    N = len(signal)                              # Length of the original signal
    K = N - lag + 1                                # Number of lagged vectors (columns in trajectory matrix)
    
    # Step 1: Embedding using multi-threading
    X = np.zeros((lag, K))                         # Initialize trajectory matrix
    threads = []                                 # List to hold thread objects
    for m in range(K):
        t = threading.Thread(target=laggingFunction, args=(m, X, signal, lag))  # Create thread for column m
        threads.append(t)                        # Add thread to list
        t.start()                                # Start the thread
    for t in threads:
        t.join()                                 # Wait for all threads to complete
    X = np.ascontiguousarray(X)                  # Ensure memory layout is contiguous for better performance
    
    # Step 2: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(fastDot(X, np.ascontiguousarray(X.T)))  # Use JIT-accelerated dot product
    idx = np.argsort(eigenvalues)[::-1]          # Sort indices in descending order
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]  # Rearrange eigenvalues and eigenvectors
    
    # Calculate principal components
    PC = fastDot(np.ascontiguousarray(eigenvectors.T), X)  # Project data onto principal components
    
    # Steps 3 & 4: Grouping and Diagonal Averaging with parallel processing
    RC = np.zeros((N, numComp), dtype=np.float64)  # Reconstructed components matrix
    for i in range(numComp):
        buff = fastBuf(eigenvectors[:, i], PC[i, :])  # Elementary matrix for i-th component
        # Parallel computation of diagonal averaging
        RC[:, i] = Parallel(n_jobs=cpu_count())(delayed(calcRC)(buff.diagonal(j)) for j in range(-buff.shape[0] + 1, buff.shape[1]))
    
    return RC, eigenvectors, eigenvalues