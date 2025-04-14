#!/usr/bin/env python
"""
Utility functions for matrix operations in seismic data processing.
"""
import logging
import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)

def create_hankel_matrix(signal, L=None):
    """
    Create a Hankel matrix from a 1D signal.
    
    A Hankel matrix has constant anti-diagonals. For a signal [a, b, c, d, e, f],
    a Hankel matrix with L=3 would be:
    
    [[a, b, c, d],
     [b, c, d, e],
     [c, d, e, f]]
    
    Parameters
    ----------
    signal : numpy.ndarray
        1D input signal
    L : int, optional
        Number of rows, by default None
        If None, uses L = len(signal) // 2
    
    Returns
    -------
    numpy.ndarray
        Hankel matrix
    """
    n = len(signal)
    
    if L is None:
        L = n // 2
        
    if L > n - 1:
        raise ValueError(f"L must be <= {n-1}, got {L}")
    
    K = n - L + 1  # Number of columns
    H = np.zeros((L, K))
    
    # Fill the Hankel matrix
    for i in range(L):
        H[i, :] = signal[i:i + K]
    
    return H

def reconstruct_from_hankel(H):
    """
    Reconstruct a 1D signal from a Hankel matrix by averaging anti-diagonals.
    
    Parameters
    ----------
    H : numpy.ndarray
        Hankel matrix
    
    Returns
    -------
    numpy.ndarray
        Reconstructed 1D signal
    """
    L, K = H.shape
    n = L + K - 1  # Length of the original signal
    
    reconstructed = np.zeros(n)
    count = np.zeros(n)
    
    # For each anti-diagonal, sum values and count elements
    for i in range(L):
        for j in range(K):
            idx = i + j  # Index in the reconstructed signal
            reconstructed[idx] += H[i, j]
            count[idx] += 1
    
    # Average by dividing by the count
    reconstructed = reconstructed / count
    
    return reconstructed

def hankel_svd_denoise(signal, rank=None, L=None):
    """
    Denoise a 1D signal using Hankel SVD.
    
    This method is also known as Singular Spectrum Analysis (SSA):
    1. Embed the signal into a Hankel matrix
    2. Perform SVD on the Hankel matrix
    3. Truncate to the specified rank
    4. Reconstruct the denoised signal by averaging anti-diagonals
    
    Parameters
    ----------
    signal : numpy.ndarray
        1D input signal
    rank : int, optional
        Number of singular values to keep, by default None
        If None, uses rank = L // 5
    L : int, optional
        Embedding dimension (window length), by default None
        If None, uses L = len(signal) // 2
    
    Returns
    -------
    numpy.ndarray
        Denoised signal
    dict
        Additional information
    """
    n = len(signal)
    
    # Set default L
    if L is None:
        L = n // 2
    
    # Set default rank
    if rank is None:
        rank = max(1, L // 5)
    
    # Create Hankel matrix
    H = create_hankel_matrix(signal, L)
    
    # Perform SVD
    U, sigma, Vt = linalg.svd(H, full_matrices=False)
    
    # Truncate to rank
    H_denoised = U[:, :rank] @ np.diag(sigma[:rank]) @ Vt[:rank, :]
    
    # Reconstruct signal
    denoised_signal = reconstruct_from_hankel(H_denoised)
    
    info = {
        'singular_values': sigma,
        'rank_used': rank,
        'embedding_dimension': L
    }
    
    return denoised_signal, info

def hankel_matrix_denoise(data, rank=None, L=None):
    """
    Apply Hankel matrix denoising to multiple traces.
    
    Parameters
    ----------
    data : numpy.ndarray
        2D array of traces (traces x samples)
    rank : int, optional
        Number of singular values to keep, by default None
    L : int, optional
        Embedding dimension, by default None
    
    Returns
    -------
    numpy.ndarray
        Denoised data
    dict
        Additional information
    """
    n_traces, n_samples = data.shape
    denoised_data = np.zeros_like(data)
    
    all_singular_values = []
    
    for i in range(n_traces):
        denoised_trace, info = hankel_svd_denoise(data[i], rank, L)
        denoised_data[i] = denoised_trace
        all_singular_values.append(info['singular_values'])
    
    # Combine information
    info = {
        'rank_used': rank,
        'embedding_dimension': L,
        'singular_values': all_singular_values
    }
    
    return denoised_data, info

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate a clean sinusoidal signal with multiple frequencies
    t = np.linspace(0, 10, 1000)
    clean_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
    
    # Add noise
    noisy_signal = clean_signal + 0.3 * np.random.randn(len(t))
    
    # Denoise using Hankel SVD
    denoised_signal, info = hankel_svd_denoise(noisy_signal, rank=5)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, clean_signal, 'b-', label='Clean Signal')
    plt.plot(t, noisy_signal, 'k-', alpha=0.4, label='Noisy Signal')
    plt.plot(t, denoised_signal, 'r-', label='Denoised Signal')
    plt.legend()
    plt.title(f"Hankel SVD Denoising (rank={info['rank_used']})")
    plt.tight_layout()
    plt.show() 