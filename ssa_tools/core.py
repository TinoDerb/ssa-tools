import numpy as np
from .classic import classic_ssa
from .accelerated import accelerated_ssa
import matplotlib.pyplot as plt

class SSA(object):
    """
    Singular Spectrum Analysis (SSA) for time series decomposition.
    
    This class implements both classic and accelerated SSA algorithms for time series analysis,
    including component reconstruction, weighted correlation analysis, and visualization tools.
    
    Attributes:
        data (np.ndarray): Input 1D time series data.
        L (int): Window length (lag) for embedding dimension.
        numComp (int): Number of components to extract in reconstruction.
        method (str): SSA method to use, either 'classic' or 'accelerated'.
        decomposition (str): Decomposition type: 'SVD' (Singular Value Decomposition) or 'EVD' (Eigenvalue Decomposition).
        K (int): Number of columns in the trajectory matrix (len(data) - L + 1).
        threshold (float): Correlation threshold for identifying related components (0.0-1.0).
        RC (np.ndarray): Reconstructed components matrix after computeSSA() is called.
        eigenvectors (np.ndarray): Eigenvectors from decomposition after computeSSA() is called.
        eigenvalues (np.ndarray): Eigenvalues from decomposition after computeSSA() is called.
        WCorr (np.ndarray): Weighted correlation matrix after computeWCA() is called.
        pairs (list): List of correlated component pairs after detectCorrelatedComponents() is called.
    """
    def __init__(self, data, lag, numComp, method="classic", decomposition="EVD", threshold=0.9):
        """
        Initializes SSA with the specified parameters.
        
        Args:
            data (np.ndarray): 1D time series data to analyze.
            lag (int): Window length, must be between 2 and len(data)//2.
            numComp (int): Number of components to extract, must be between 1 and lag.
            method (str, optional): Algorithm to use ('classic' or 'accelerated'). Defaults to "classic".
            decomposition (str, optional): Decomposition method ('SVD' or 'EVD'). Defaults to "EVD".
            threshold (float, optional): Correlation threshold for component grouping. Defaults to 0.9.
        
        Raises:
            AssertionError: If any of the input parameters don't meet requirements.
        """
        assert isinstance(data, np.ndarray) and data.ndim == 1, "data must be 1D and of type ndarray"  # Validate data shape
        assert isinstance(lag, int) and 2 <= lag < len(data) // 2, "Lag must be between 2 and len(data)//2, and an integer"  # Validate lag
        assert isinstance(numComp, int) and 1 <= numComp <= lag, "numComp must be between 1 and lag, and an integer"  # Validate number of components
        assert method in ["classic", "accelerated"], "method can only be 'classic' or 'accelerated'"  # Validate method
        assert decomposition in ["SVD", "EVD"], "decomposition can only be 'SVD' or 'EVD'"  # Validate decomposition
        
        self.data = data                         # Store time series data
        self.L = lag                             # Store window length
        self.numComp = numComp                   # Store number of components to extract
        self.method = method                     # Store selected algorithm method
        self.decomposition = decomposition       # Store decomposition type
        self.K = len(data) - self.L + 1          # Compute trajectory matrix width (number of lagged vectors)
        self.threshold = threshold               # Store correlation threshold for component grouping
        
        # These attributes will be set by other methods:
        # self.RC - Reconstructed components
        # self.eigenvectors - Eigenvectors from decomposition
        # self.eigenvalues - Eigenvalues from decomposition
        # self.WCorr - Weighted correlation matrix
        # self.pairs - List of correlated component pairs

    def computeSSA(self):
        """
        Computes the SSA decomposition and stores the reconstructed components, eigenvectors, and eigenvalues.
        
        This method performs the core SSA algorithm, either using the classic or accelerated implementation.
        After calling, the following attributes will be available:
        - self.RC: Reconstructed components matrix
        - self.eigenvectors: Eigenvectors from the decomposition
        - self.eigenvalues: Eigenvalues from the decomposition
        
        Raises:
            AssertionError: If accelerated method is used with SVD decomposition (not supported).
        """
        if self.method == "accelerated":          # Use accelerated SSA if selected
            assert self.decomposition == "EVD", "Only decomposition = 'EVD' is supported while choosing method = 'accelerated'"
            self.RC, self.eigenvectors, self.eigenvalues = accelerated_ssa(self.data, self.L, self.numComp)  # Call optimized implementation
        else:                                     # Use classic SSA otherwise
            self.RC, self.eigenvectors, self.eigenvalues = classic_ssa(self.data, self.L, self.numComp, self.decomposition)  # Call standard implementation

    def computeWCA(self):
        """
        Computes the weighted correlation (WCorr) matrix between reconstructed components.
        
        This method calculates pairwise correlations between components taking into account the 
        structure of the time series. High correlation suggests components might be grouped.
        After calling, self.WCorr will contain the weighted correlation matrix.
        
        Raises:
            AssertionError: If called before computeSSA().
            
        Notes:
            Code adapted from: https://www.kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition
        """
        assert hasattr(self, "RC"), "Cannot execute computeWCA() before computeSSA()"  # Ensure SSA has been computed
        
        L = self.L                               # Window length
        K = self.K                               # Trajectory matrix width
        RC = self.RC                             # Reconstructed components
        numComp = self.numComp                   # Number of components
        
        def w_inner(F_i, F_j, w):                # Define weighted inner product function
            return np.dot(w * F_i, F_j)          # Compute inner product with weights
        
        # Construct trapezoidal weight vector based on number of terms in each lag
        w = np.array(list(np.arange(L)+1) + [L] * (K - L - 1) + list(np.arange(L)+1)[::-1])
        
        # Compute weighted norms for each component
        F_wnorms = np.array([w_inner(RC[:, i], RC[:, i], w) for i in range(numComp)])
        F_wnorms = F_wnorms ** -0.5              # Invert and take square root for normalization
        
        # Initialize and compute the weighted correlation matrix
        WCorr = np.identity(numComp)             # Start with identity matrix (diagonal is 1)
        for i in range(numComp):
            for j in range(i + 1, numComp):      # Only compute upper triangle (matrix is symmetric)
                WCorr[i, j] = abs(w_inner(RC[:, i], RC[:, j], w) * F_wnorms[i] * F_wnorms[j])  # Upper triangle
                WCorr[j, i] = WCorr[i, j]        # Copy value to lower triangle (symmetric matrix)
                
        self.WCorr = WCorr                       # Store weighted correlation matrix as attribute

    def plotReconstructedComponents(self):
        """
        Plots the reconstructed components of the time series.
        
        This method visualizes the original time series alongside its reconstructed components,
        allowing for visual inspection of the decomposition results.
        
        Raises:
            AssertionError: If called before computeSSA().
        """
        assert hasattr(self, "RC"), "Cannot plot reconstructed components before computeSSA()"  # Ensure SSA is done
        
        RC = self.RC                             # Get components matrix
        data = self.data                         # Get original data
        
        plt.figure(figsize=(6, 6))               # Set figure size
        plt.plot(data, color='k', linewidth=3, label="Original")  # Plot original data in black
        
        for i in range(RC.shape[1]):
            if i < 5:                            # Only include first 5 components in legend
                plt.plot(RC[:, i], label=f"Component {i}")  # Plot each RC with label
            else:
                plt.plot(RC[:, i], label='_nolegend_')  # Plot each RC without label
                
        plt.title("Reconstructed components", fontsize=14, fontname="Arial")  # Add title
        plt.xlabel("Sample number", fontsize=14, fontname="Arial")  # X label
        plt.ylabel("Value", fontsize=14, fontname="Arial")  # Y label
        plt.legend(loc="upper left")             # Add legend
        plt.tight_layout()                       # Improve layout

    def plotVarianceExplained(self):
        """
        Plots the variance explained and cumulative variance explained by each component.
        
        This method creates a dual-axis plot showing:
        1. Individual variance explained by each component
        2. Cumulative variance explained up to each component
        
        This visualization helps identify how many components are needed to capture
        a significant portion of the signal.
        
        Raises:
            AssertionError: If called before computeSSA().
        """
        assert hasattr(self, "eigenvalues"), "Cannot plot variance explained before computeSSA()"  # Ensure SSA is done
        
        varianceExplained = 100 * self.eigenvalues / self.eigenvalues.sum()  # Compute percentage variance per component
        
        plt.figure(figsize=(6, 6))               # Set figure size
        plt.plot(varianceExplained, "o-", label="Explained")  # Plot individual explained variance
        plt.xlabel("Component number", fontsize=14, fontname="Arial")  # X label
        plt.ylabel("Variance explained (%)", fontsize=14, fontname="Arial")  # Y label (left)
        
        plt.twinx()                              # Add secondary Y axis
        plt.plot(np.cumsum(varianceExplained), "r--o", label="Cumulative")  # Plot cumulative variance
        plt.ylabel("Cumulative variance explained (%)", fontsize=14, fontname="Arial")  # Y label (right)
        
        plt.title("Component variance analysis", fontsize=14, fontname="Arial")  # Title
        plt.tight_layout()                       # Adjust layout

    def plotWeightedCorrelations(self):
        """
        Displays a heatmap of the weighted correlations between reconstructed components.
        
        This visualization shows which components are highly correlated and might represent
        the same underlying pattern or oscillation in the time series.
        Components with high correlation (close to 1.0) may be grouped together.
        
        Raises:
            AssertionError: If called before computeWCA().
        """
        assert hasattr(self, "WCorr"), "Cannot plot WCorr before computeWCA()"  # Ensure WCorr is computed
        
        WCorr = self.WCorr                       # Get WCorr matrix
        
        plt.figure(figsize=(8.5, 7))             # Set figure size
        plt.imshow(WCorr, cmap='viridis', vmin=0, vmax=1)  # Show WCorr as heatmap with color scale 0-1
        
        plt.xlabel("Component number", fontsize=14, fontname="Arial")  # X label
        plt.ylabel("Component number", fontsize=14, fontname="Arial")  # Y label
        
        for i in range(WCorr.shape[0]):          # Add grid lines to separate components
            plt.axvline(i+0.5, color="k", linestyle="--")  # Vertical grid line
            plt.axhline(i+0.5, color="k", linestyle="--")  # Horizontal grid line
            
        cbar = plt.colorbar()                    # Add color scale bar
        cbar.set_label(label='Weighted Correlation', font="Arial", size=14)  # Set colorbar label
        
        plt.title("Weighted correlations", fontsize=14, fontname="Arial")  # Title
        plt.tight_layout()                       # Adjust layout

    def detectCorrelatedComponents(self, listPairs=False):
        """
        Detects pairs of components with high weighted correlation.
        
        This method identifies components that are likely representing the same
        underlying pattern based on their weighted correlation exceeding the threshold.
        
        Args:
            listPairs (bool, optional): If True, prints out correlated component pairs. Defaults to False.
        
        After calling, the following attributes will be available:
        - self.pairs: indices of reconstructed components that can be grouped together due to their high correlation
            
        Raises:
            AssertionError: If called before computeWCA().
        """
        assert hasattr(self, "WCorr"), "Cannot detect correlations before computeWCA()"  # Ensure WCorr is available
        
        WCorr = self.WCorr                       # Get WCorr matrix
        numComp = self.numComp                   # Get number of components
        pairs = []                               # List to hold correlated pairs
        
        for i in range(numComp):                 # Loop through components
            for j in range(i + 1, numComp):      # Check each unique pair once (upper triangle)
                if abs(WCorr[i, j]) > self.threshold:  # Check against correlation threshold
                    if listPairs:                # If requested to print results
                        print(f"Components {i} and {j} are highly correlated (R = {WCorr[i, j]:.2f})")  # Print pair info
                    pairs.append((i, j))         # Store correlated pair
                    
        self.pairs = pairs                       # Save pairs as attribute