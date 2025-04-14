#!/usr/bin/env python
"""
Preprocessing functions for seismic data.
"""
import logging
import numpy as np
from scipy import signal
from obspy.core import Stream
from obspy.signal.filter import bandpass

logger = logging.getLogger(__name__)

def preprocess_seismic_data(data, sampling_rate=None):
    """
    Preprocess seismic data before denoising.
    
    Parameters
    ----------
    data : obspy.core.Stream or dict
        Seismic data to preprocess
    sampling_rate : float, optional
        Sampling rate in Hz, by default None
        
    Returns
    -------
    dict
        Preprocessed data
    """
    # Handle different input types
    if isinstance(data, Stream):
        return _preprocess_stream(data)
    elif isinstance(data, dict):
        if sampling_rate is None and 'sampling_rate' in data:
            sampling_rate = data['sampling_rate']
        return _preprocess_numpy(data, sampling_rate)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

def _preprocess_stream(stream):
    """
    Preprocess ObsPy Stream object.
    
    Parameters
    ----------
    stream : obspy.core.Stream
        Seismic data stream
        
    Returns
    -------
    dict
        Preprocessed data
    """
    logger.info(f"Preprocessing ObsPy Stream with {len(stream)} traces")
    
    # Copy to avoid modifying the original
    stream_copy = stream.copy()
    
    # Basic preprocessing for each trace
    for trace in stream_copy:
        # Remove mean
        trace.detrend('demean')
        
        # Remove linear trend
        trace.detrend('linear')
        
        # Apply bandpass filter
        trace.filter('bandpass', freqmin=1.0, freqmax=20.0)
        
        # Taper edges to reduce edge effects
        trace.taper(max_percentage=0.05, type='hann')
    
    # Extract data matrix for processing
    # Make sure all traces have the same length
    min_len = min(len(tr.data) for tr in stream_copy)
    data_matrix = np.vstack([tr.data[:min_len] for tr in stream_copy])
    
    sampling_rate = stream_copy[0].stats.sampling_rate
    
    return {
        'data': data_matrix,
        'sampling_rate': sampling_rate,
        'original_stream': stream
    }

def _preprocess_numpy(data_dict, sampling_rate=100):
    """
    Preprocess data in numpy array format.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing seismic data arrays
    sampling_rate : float, optional
        Sampling rate in Hz, by default 100
        
    Returns
    -------
    dict
        Preprocessed data
    """
    logger.info("Preprocessing numpy array data")
    
    # Extract data
    if 'noisy_data' in data_dict:
        input_data = data_dict['noisy_data'].copy()
    else:
        input_data = data_dict['data'].copy() if 'data' in data_dict else next(iter(data_dict.values())).copy()
    
    n_traces, n_samples = input_data.shape
    logger.info(f"Data shape: {input_data.shape}")
    
    # Normalize each trace
    processed_data = np.zeros_like(input_data)
    for i in range(n_traces):
        trace = input_data[i, :]
        
        # Remove mean
        trace = trace - np.mean(trace)
        
        # Remove trend
        trace = signal.detrend(trace)
        
        # Apply bandpass filter (1-20 Hz typical for seismic)
        nyquist = sampling_rate / 2
        low, high = 1.0 / nyquist, 20.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        trace = signal.filtfilt(b, a, trace)
        
        # Normalize
        if np.std(trace) > 0:
            trace = trace / np.std(trace)
        
        processed_data[i, :] = trace
    
    result = {
        'data': processed_data,
        'sampling_rate': sampling_rate
    }
    
    # Keep original data if available
    if 'noisy_data' in data_dict:
        result['noisy_data'] = data_dict['noisy_data']
    if 'clean_data' in data_dict:
        result['clean_data'] = data_dict['clean_data']
    
    return result

# Example usage for testing
if __name__ == "__main__":
    from src.data.dataset import SeismicDataLoader
    
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    loader = SeismicDataLoader("data/raw")
    data = loader.load_data()
    
    # Preprocess it
    processed = preprocess_seismic_data(data)
    print(f"Preprocessed data shape: {processed['data'].shape}") 