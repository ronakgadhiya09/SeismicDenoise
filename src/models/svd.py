#!/usr/bin/env python
"""
SVD-based denoising for seismic data.
"""
import logging
import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)

class SVDDenoiser:
    """
    Class for denoising seismic data using Singular Value Decomposition.
    
    This implements a basic form of low-rank approximation for noise reduction,
    assuming that the signal lies in a low-dimensional subspace while noise
    is spread across all dimensions.
    """
    
    def __init__(self, rank=None, threshold=None, fraction=None):
        """
        Initialize the SVD denoiser.
        
        Parameters
        ----------
        rank : int, optional
            Fixed rank to use for approximation, by default None
        threshold : float, optional
            Threshold for singular values (values below threshold*max_sigma are set to zero), by default None
        fraction : float, optional
            Fraction of energy to preserve (0.0-1.0), by default None
            
        Notes
        -----
        Exactly one of rank, threshold, or fraction should be specified.
        If none is specified, rank=10 is used by default.
        """
        self.rank = rank
        self.threshold = threshold
        self.fraction = fraction
        
        # Default to rank-based if none specified
        if rank is None and threshold is None and fraction is None:
            self.rank = 10
            
        if sum(x is not None for x in [rank, threshold, fraction]) > 1:
            raise ValueError("Only one of rank, threshold, or fraction should be specified")
    
    def denoise(self, data):
        """
        Denoise data using SVD-based low-rank approximation.
        
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
        
        # Compute SVD
        logger.info("Computing SVD...")
        U, sigma, Vt = linalg.svd(data_matrix, full_matrices=False)
        
        logger.info(f"Singular values: {sigma[:10]}...")
        
        # Determine number of components to keep
        if self.rank is not None:
            k = min(self.rank, len(sigma))
            logger.info(f"Using fixed rank: {k}")
        elif self.threshold is not None:
            k = np.sum(sigma > self.threshold * sigma[0])
            logger.info(f"Using threshold {self.threshold}, keeping {k} components")
        elif self.fraction is not None:
            total_energy = np.sum(sigma**2)
            cumulative_energy = np.cumsum(sigma**2) / total_energy
            k = np.searchsorted(cumulative_energy, self.fraction) + 1
            logger.info(f"Using energy fraction {self.fraction}, keeping {k} components")
            
        # Generate low-rank approximation
        logger.info(f"Reconstructing with {k} components...")
        denoised_matrix = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
        
        # Prepare output
        output = input_dict.copy()
        output['denoised_data'] = denoised_matrix
        output['singular_values'] = sigma
        output['rank_used'] = k
        
        return output
        
    def _truncate_svd(self, U, sigma, Vt, k):
        """Truncate SVD components to top-k."""
        return U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]

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
    denoiser = SVDDenoiser(rank=5)
    result = denoiser.denoise(processed)
    
    # Compare (use original clean data as reference if available)
    if 'clean_data' in data:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        trace_idx = 5  # Show the 5th trace
        
        axes[0].plot(data['clean_data'][trace_idx])
        axes[0].set_title('Original Clean Signal')
        
        axes[1].plot(data['noisy_data'][trace_idx])
        axes[1].set_title('Noisy Signal')
        
        axes[2].plot(result['denoised_data'][trace_idx])
        axes[2].set_title(f'Denoised Signal (rank={denoiser.rank})')
        
        plt.tight_layout()
        plt.show() 