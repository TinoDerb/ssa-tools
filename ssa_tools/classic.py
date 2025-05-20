import numpy as np

<<<<<<< HEAD
def classic_ssa(signal, lag, numComp, decomposition):
    '''
    Performs Classic Singular Spectrum Analysis (SSA) on a time series signal.
    
    This function implements the four steps of SSA:
    1. Embedding: Convert the 1D time series into a trajectory matrix
    2. Decomposition: Apply SVD or eigenvalue decomposition to the trajectory matrix
    3. Grouping: Select components to reconstruct (based on numComp parameter)
    4. Diagonal averaging: Convert matrices back to time series format
    
    Parameters:
        signal (array-like): The input time series signal to decompose
        lag (int): Window length for embedding (must be: 2 <= lag <= N/2)
        numComp (int): Number of components to reconstruct (must be <= lag)
        decomposition (str): Method for decomposition, either "SVD" or "eigenvalue"
    
    Returns:
        tuple:
            - RC (np.ndarray): Reconstructed components as columns, shape (N, numComp)
            - eigenvectors (np.ndarray): The eigenvectors from decomposition
            - eigenvalues (np.ndarray): The eigenvalues from decomposition
    
    Reference:
        Based on: https://www.kaggle.code/jdarcy/introducing-ssa-for-time-series-decomposition
    '''
    N = len(signal)                     # Length of the original signal
    K = N - lag + 1                     # Number of lagged vectors (columns in trajectory matrix)
    L = lag                             # Define L instead of lag for nicer text
    # Step 1: Embedding - Create trajectory matrix by sliding window
    X = np.array([signal[i:L+i] for i in range(0, K)]).T  # Trajectory matrix of shape (L, K)
    
    if decomposition == "SVD":
        # Step 2: Decomposition using Singular Value Decomposition
        U, Sigma, VT = np.linalg.svd(X)          # Apply SVD to trajectory matrix
        eigenvectors = U                         # The left singular vectors
        eigenvalues = Sigma**2                   # Square of singular values equals eigenvalues
        
        # Steps 3 & 4: Grouping and Diagonal Averaging
        RC = np.zeros((N, numComp))              # Reconstructed components matrix
        for i in range(numComp):
            # Elementary matrix for i-th component
            buff = Sigma[i] * np.outer(U[:, i], VT[i, :])  
            buff = buff[::-1]                    # Flip for anti-diagonal averaging
            # Diagonal averaging - compute mean along anti-diagonals
            RC[:, i] = [buff.diagonal(j).mean() for j in range(-buff.shape[0] + 1, buff.shape[1])]
            
    else:
        # Step 2: Decomposition using Eigenvalue Decomposition of lag-covariance matrix
        C = X @ X.T                              # Compute lag-covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(C)  # Eigendecomposition (symmetric matrix)
        idx = np.argsort(eigenvalues)[::-1]      # Sort indices in descending order
        eigenvalues = eigenvalues[idx]           # Rearrange eigenvalues
        eigenvectors = eigenvectors[:, idx]      # Rearrange eigenvectors
        
        PC = eigenvectors.T @ X                  # Project data onto principal components
        
        # Steps 3 & 4: Grouping and Diagonal Averaging
        RC = np.zeros((N, numComp))              # Reconstructed components matrix
        for i in range(numComp):
            buff = np.outer(eigenvectors[:, i], PC[i, :])  # Elementary matrix for i-th component
            buff = buff[::-1]                    # Flip for anti-diagonal averaging
            # Diagonal averaging - compute mean along anti-diagonals
            RC[:, i] = [buff.diagonal(j).mean() for j in range(-buff.shape[0] + 1, buff.shape[1])]
            
    return RC, eigenvectors, eigenvalues
=======
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
>>>>>>> cdc42936f83c2f0aba3cc6f07367ab6063b26bb0
