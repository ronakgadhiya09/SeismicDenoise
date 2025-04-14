#!/usr/bin/env python
"""
Basic tests for the denoising algorithms.
"""
import os
import sys
import unittest
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import SeismicDataLoader
from src.preprocessing.preprocess import preprocess_seismic_data
from src.models.svd import SVDDenoiser
from src.models.rpca import RPCADenoiser
from src.models.hankel import HankelDenoiser

class TestDenoising(unittest.TestCase):
    """Test case for denoising algorithms."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic test data directly
        np.random.seed(42)  # For reproducibility
        
        # Time vector
        t = np.linspace(0, 10, 1000)
        
        # Create clean signal (Ricker wavelets)
        self.n_traces = 5
        self.n_samples = 1000
        self.clean_data = np.zeros((self.n_traces, self.n_samples))
        
        for i in range(self.n_traces):
            # Varying central frequency
            freq = 3 + i * 0.5
            
            # Ricker wavelet
            wavelet = (1 - 2 * (np.pi * freq * (t - 5))**2) * np.exp(-(np.pi * freq * (t - 5))**2)
            
            # Add some shift
            shift = int(self.n_samples * 0.1 * i / self.n_traces)
            rolled_wavelet = np.roll(wavelet, shift)
            
            # Add signal
            self.clean_data[i, :] = rolled_wavelet
        
        # Add noise
        noise_level = 0.2
        noise = noise_level * np.random.randn(self.n_traces, self.n_samples)
        self.noisy_data = self.clean_data + noise
        
        # Create test data dictionary
        self.test_data = {
            'clean_data': self.clean_data,
            'noisy_data': self.noisy_data,
            'sampling_rate': 100
        }
        
        # Preprocess data
        self.preprocessed_data = preprocess_seismic_data(self.test_data)
    
    def test_svd_denoiser(self):
        """Test SVD denoiser."""
        denoiser = SVDDenoiser(rank=3)
        result = denoiser.denoise(self.preprocessed_data)
        
        # Check that result has expected keys
        self.assertIn('denoised_data', result)
        self.assertIn('singular_values', result)
        self.assertIn('rank_used', result)
        
        # Check dimensions
        self.assertEqual(result['denoised_data'].shape, self.noisy_data.shape)
        
        # Rank should be 3
        self.assertEqual(result['rank_used'], 3)
        
        # Denoised data should be closer to clean data than noisy data
        mse_noisy = np.mean((self.clean_data - self.noisy_data)**2)
        mse_denoised = np.mean((self.clean_data - result['denoised_data'])**2)
        self.assertLess(mse_denoised, mse_noisy)
    
    def test_rpca_denoiser(self):
        """Test RPCA denoiser."""
        denoiser = RPCADenoiser(rank=3, max_iter=10)  # Limit iterations for faster test
        result = denoiser.denoise(self.preprocessed_data)
        
        # Check that result has expected keys
        self.assertIn('denoised_data', result)
        self.assertIn('noise_component', result)
        
        # Check dimensions
        self.assertEqual(result['denoised_data'].shape, self.noisy_data.shape)
        self.assertEqual(result['noise_component'].shape, self.noisy_data.shape)
        
        # Denoised + noise should approximate original
        reconstructed = result['denoised_data'] + result['noise_component']
        np.testing.assert_allclose(reconstructed, self.preprocessed_data['data'], rtol=1e-5)
    
    def test_hankel_denoiser(self):
        """Test Hankel SVD denoiser."""
        denoiser = HankelDenoiser(rank=3)
        result = denoiser.denoise(self.preprocessed_data)
        
        # Check that result has expected keys
        self.assertIn('denoised_data', result)
        self.assertIn('rank_used', result)
        self.assertIn('embedding_dimension', result)
        
        # Check dimensions
        self.assertEqual(result['denoised_data'].shape, self.noisy_data.shape)
        
        # Rank should be 3
        self.assertEqual(result['rank_used'], 3)

if __name__ == '__main__':
    unittest.main() 