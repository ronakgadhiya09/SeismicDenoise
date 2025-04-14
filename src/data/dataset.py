#!/usr/bin/env python
"""
Seismic data loading and handling.
"""
import os
import logging
import glob
import numpy as np
from obspy import read
from obspy.core import Stream

logger = logging.getLogger(__name__)

class SeismicDataLoader:
    """Class for loading and processing seismic data."""
    
    def __init__(self, data_dir):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        data_dir : str
            Directory containing seismic data files.
        """
        self.data_dir = data_dir
        
    def load_data(self, file_pattern="*.mseed"):
        """
        Load seismic data from files.
        
        Parameters
        ----------
        file_pattern : str, optional
            Pattern to match seismic data files, by default "*.mseed"
            
        Returns
        -------
        obspy.core.Stream or numpy.ndarray
            Loaded seismic data
        """
        # Search for files
        file_paths = sorted(glob.glob(os.path.join(self.data_dir, file_pattern)))
        
        if not file_paths:
            # If no real data files, create synthetic data for demonstration
            logger.warning(f"No seismic data files found in {self.data_dir}. Using synthetic data.")
            return self._create_synthetic_data()
        
        # Load real seismic data
        logger.info(f"Loading {len(file_paths)} seismic data files")
        stream = Stream()
        for file_path in file_paths:
            try:
                st = read(file_path)
                stream += st
                logger.debug(f"Loaded {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return stream
    
    def _create_synthetic_data(self, n_traces=10, n_samples=1000):
        """
        Create synthetic seismic data for demonstration purposes.
        
        Parameters
        ----------
        n_traces : int, optional
            Number of seismic traces, by default 10
        n_samples : int, optional
            Number of samples per trace, by default 1000
            
        Returns
        -------
        numpy.ndarray
            Synthetic seismic data matrix of shape (n_traces, n_samples)
        """
        logger.info(f"Creating synthetic data with {n_traces} traces of {n_samples} samples each")
        
        # Time vector
        t = np.linspace(0, 10, n_samples)
        
        # Create clean signal (Ricker wavelets with varying frequencies)
        data = np.zeros((n_traces, n_samples))
        for i in range(n_traces):
            # Varying central frequency
            freq = 5 + i * 0.5
            
            # Ricker wavelet (Mexican hat)
            # More realistic for seismic data
            sigma = 1.0
            wavelet = (1 - 2 * (np.pi * freq * (t - 5))**2) * np.exp(-(np.pi * freq * (t - 5))**2)
            
            # Add some temporal shift for each trace
            shift = int(n_samples * 0.2 * i / n_traces)
            rolled_wavelet = np.roll(wavelet, shift)
            
            # Add signal
            data[i, :] = rolled_wavelet
        
        # Add noise
        noise_level = 0.3
        noise = noise_level * np.random.randn(n_traces, n_samples)
        noisy_data = data + noise
        
        return {
            'noisy_data': noisy_data,
            'clean_data': data,
            'sampling_rate': 100  # Hz
        }
    
    def save_data(self, data, output_file):
        """
        Save processed data to file.
        
        Parameters
        ----------
        data : dict or numpy.ndarray
            Data to save
        output_file : str
            Output file path
        """
        try:
            if isinstance(data, dict):
                np.savez(output_file, **data)
            else:
                np.savez(output_file, data=data)
            logger.info(f"Data saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving data to {output_file}: {e}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = SeismicDataLoader("data/raw")
    data = loader.load_data()
    print(f"Generated synthetic data shape: {data['noisy_data'].shape}") 