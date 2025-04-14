#!/usr/bin/env python
"""
Hankel matrix SVD denoising for seismic data.
"""
import logging
import numpy as np
from src.utils.matrix_utils import hankel_matrix_denoise

logger = logging.getLogger(__name__)

class HankelDenoiser:
    """
    Class for denoising seismic data using Hankel matrix SVD.
    
    This technique, also known as Singular Spectrum Analysis (SSA),
    embeds time series into a Hankel matrix and performs low-rank
    approximation in that domain. It's particularly effective for
    seismic data that contains oscillatory patterns.
    """
    
    def __init__(self, rank=None, window_length=None):
        """
        Initialize the Hankel SVD denoiser.
        
        Parameters
        ----------
        rank : int, optional
            Number of singular values to keep, by default None.
            If None, will be determined adaptively based on window_length.
        window_length : int, optional
            Embedding dimension (window length) for the Hankel matrix, by default None.
            If None, will be determined adaptively based on data length.
        """
        self.rank = rank
        self.window_length = window_length
    
    def denoise(self, data):
        """
        Denoise data using Hankel matrix SVD.
        
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
        logger.info(f"Denoising data matrix of shape {data_matrix.shape} using Hankel SVD")
        
        # Apply Hankel matrix denoising
        denoised_data, info = hankel_matrix_denoise(
            data_matrix, 
            rank=self.rank, 
            L=self.window_length
        )
        
        # Prepare output
        output = input_dict.copy()
        output['denoised_data'] = denoised_data
        output['rank_used'] = info['rank_used']
        output['embedding_dimension'] = info['embedding_dimension']
        output['singular_values'] = info['singular_values'][0]  # Use first trace's values
        
        return output

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
    denoiser = HankelDenoiser(rank=5)
    result = denoiser.denoise(processed)
    
    # Compare
    if 'clean_data' in data:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        trace_idx = 5  # Show the 5th trace
        
        axes[0].plot(data['clean_data'][trace_idx])
        axes[0].set_title('Original Clean Signal')
        
        axes[1].plot(data['noisy_data'][trace_idx])
        axes[1].set_title('Noisy Signal')
        
        axes[2].plot(result['denoised_data'][trace_idx])
        axes[2].set_title(f'Denoised Signal (Hankel SVD, rank={result["rank_used"]})')
        
        plt.tight_layout()
        plt.show() 