#!/usr/bin/env python
"""
Robust Principal Component Analysis (RPCA) for seismic data denoising.
"""
import logging
import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)

class RPCADenoiser:
    """
    Class for denoising seismic data using Robust Principal Component Analysis.
    
    RPCA decomposes a matrix M into a low-rank component L and a sparse component S:
    M = L + S
    
    This is particularly useful for seismic data where:
    - L captures the coherent signal (low-rank structure)
    - S captures spikes, outliers, and random noise (sparse component)
    
    This implementation uses the Principal Component Pursuit (PCP) algorithm via
    the Alternating Direction Method of Multipliers (ADMM).
    """
    
    def __init__(self, rank=10, lmbda=None, max_iter=100, tol=1e-7):
        """
        Initialize the RPCA denoiser.
        
        Parameters
        ----------
        rank : int, optional
            Target rank for the low-rank component, by default 10
        lmbda : float, optional
            Regularization parameter. If None, uses 1/sqrt(max(dimensions))
        max_iter : int, optional
            Maximum number of iterations, by default 100
        tol : float, optional
            Convergence tolerance, by default 1e-7
        """
        self.rank = rank
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.tol = tol
    
    def denoise(self, data):
        """
        Denoise data using RPCA.
        
        Parameters
        ----------
        data : dict or numpy.ndarray
            Data to denoise, either a dict with 'data' key or a numpy array
            
        Returns
        -------
        dict
            Denoised data and additional information
        """
        # Extract data matrix
        if isinstance(data, dict):
            data_matrix = data['data']
            input_dict = data.copy()
        else:
            data_matrix = data
            input_dict = {'data': data_matrix}
        
        # Get matrix dimensions
        n_traces, n_samples = data_matrix.shape
        logger.info(f"Denoising data matrix of shape {data_matrix.shape}")
        
        # Run RPCA
        low_rank, sparse, n_iter = self._rpca(data_matrix)
        
        # Prepare output
        output = input_dict.copy()
        output['denoised_data'] = low_rank  # The low-rank component is the denoised signal
        output['noise_component'] = sparse  # The sparse component is the noise
        output['n_iterations'] = n_iter
        
        return output
    
    def _rpca(self, M):
        """
        Implements Robust PCA via Principal Component Pursuit (PCP).
        
        Parameters
        ----------
        M : numpy.ndarray
            Input matrix to decompose
            
        Returns
        -------
        L : numpy.ndarray
            Low-rank component
        S : numpy.ndarray
            Sparse component
        n_iter : int
            Number of iterations performed
        """
        # Get matrix dimensions
        m, n = M.shape
        
        # Initialize default lambda if not specified
        if self.lmbda is None:
            lmbda = 1.0 / np.sqrt(max(m, n))
        else:
            lmbda = self.lmbda
            
        logger.info(f"Running RPCA with lambda={lmbda}, max_iter={self.max_iter}")
        
        # Initialize variables
        L = np.zeros_like(M)
        S = np.zeros_like(M)
        Y = np.zeros_like(M)  # Lagrange multiplier
        mu = 1.25 / np.linalg.norm(M)  # Step size parameter
        mu_bar = mu * 1e7
        rho = 1.5  # ADMM step size adjustment
        
        # ADMM iterations
        for i in range(self.max_iter):
            # Update L with SVD soft thresholding
            U, sigma, Vt = linalg.svd(M - S + Y/mu, full_matrices=False)
            sigma_threshold = self._svd_threshold(sigma, 1/mu)
            r = min(self.rank, len(sigma_threshold))
            L_new = U[:, :r] @ np.diag(sigma_threshold[:r]) @ Vt[:r, :]
            
            # Update S with element-wise soft thresholding
            S_new = self._soft_threshold(M - L_new + Y/mu, lmbda/mu)
            
            # Update Y (Lagrange multiplier)
            Z = M - L_new - S_new
            Y = Y + mu * Z
            
            # Check convergence
            primal_error = np.linalg.norm(Z, 'fro') / max(np.linalg.norm(M, 'fro'), 1)
            dual_error = mu * np.linalg.norm(S_new - S, 'fro') / max(np.linalg.norm(Y, 'fro'), 1)
            
            # Update L and S
            L, S = L_new, S_new
            
            # Adjust mu
            mu = min(mu_bar, rho * mu)
            
            # Log progress
            if (i+1) % 10 == 0:
                logger.info(f"Iteration {i+1}: primal error = {primal_error:.6f}, dual error = {dual_error:.6f}")
            
            # Check convergence
            if primal_error < self.tol and dual_error < self.tol:
                logger.info(f"Converged after {i+1} iterations")
                break
        
        return L, S, i+1
    
    def _svd_threshold(self, sigma, tau):
        """Apply soft thresholding to singular values."""
        return np.maximum(sigma - tau, 0)
    
    def _soft_threshold(self, X, tau):
        """Apply element-wise soft thresholding."""
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

# Example usage
if __name__ == "__main__":
    from src.data.dataset import SeismicDataLoader
    from src.preprocessing.preprocess import preprocess_seismic_data
    import matplotlib.pyplot as plt
    
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    loader = SeismicDataLoader("data/raw")
    data = loader.load_data()
    
    # Preprocess
    processed = preprocess_seismic_data(data)
    
    # Denoise
    denoiser = RPCADenoiser(rank=5)
    result = denoiser.denoise(processed)
    
    # Compare
    if 'clean_data' in data:
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        trace_idx = 5  # Show the 5th trace
        
        axes[0].plot(data['clean_data'][trace_idx])
        axes[0].set_title('Original Clean Signal')
        
        axes[1].plot(data['noisy_data'][trace_idx])
        axes[1].set_title('Noisy Signal')
        
        axes[2].plot(result['denoised_data'][trace_idx])
        axes[2].set_title('Denoised Signal (Low-rank component)')
        
        axes[3].plot(result['noise_component'][trace_idx])
        axes[3].set_title('Extracted Noise (Sparse component)')
        
        plt.tight_layout()
        plt.show() 