#!/usr/bin/env python
"""
Visualization functions for seismic data and denoising results.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

logger = logging.getLogger(__name__)

# Define a custom colormap for seismic data
SEISMIC_CMAP = LinearSegmentedColormap.from_list(
    'seismic_custom', 
    ['#0000FF', '#FFFFFF', '#FF0000'],
    N=256
)

def plot_comparison(raw_data, preprocessed_data, denoised_data, output_file=None):
    """
    Plot comparison between raw, preprocessed, and denoised data.
    
    Parameters
    ----------
    raw_data : dict
        Raw seismic data
    preprocessed_data : dict
        Preprocessed seismic data
    denoised_data : dict
        Denoised seismic data
    output_file : str, optional
        Path to save figure, by default None
    """
    logger.info("Generating comparison plot")
    
    # Extract data matrices
    if 'noisy_data' in raw_data:
        raw_matrix = raw_data['noisy_data']
    else:
        raw_matrix = raw_data['data'] if 'data' in raw_data else None
    
    preprocessed_matrix = preprocessed_data['data']
    denoised_matrix = denoised_data['denoised_data']
    
    # If raw_matrix is None, use preprocessed_matrix for display
    if raw_matrix is None:
        logger.warning("Raw data matrix not found, using preprocessed data for raw visualization")
        raw_matrix = preprocessed_matrix
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])
    
    # Top row: Trace displays for a single example trace
    trace_idx = min(5, raw_matrix.shape[0] - 1)  # Use 5th trace or last if fewer
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(raw_matrix[trace_idx], 'k-')
    ax1.set_title('Raw Trace')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, raw_matrix.shape[1])
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(preprocessed_matrix[trace_idx], 'k-')
    ax2.set_title('Preprocessed Trace')
    ax2.set_xlim(0, preprocessed_matrix.shape[1])
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(denoised_matrix[trace_idx], 'k-')
    ax3.set_title('Denoised Trace')
    ax3.set_xlim(0, denoised_matrix.shape[1])
    
    # Middle row: 2D displays (traces vs time)
    vmin = np.min([np.min(raw_matrix), np.min(preprocessed_matrix), np.min(denoised_matrix)])
    vmax = np.max([np.max(raw_matrix), np.max(preprocessed_matrix), np.max(denoised_matrix)])
    
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(raw_matrix, aspect='auto', cmap=SEISMIC_CMAP, vmin=vmin, vmax=vmax)
    ax4.set_title('Raw Data')
    ax4.set_ylabel('Trace Number')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(preprocessed_matrix, aspect='auto', cmap=SEISMIC_CMAP, vmin=vmin, vmax=vmax)
    ax5.set_title('Preprocessed Data')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(denoised_matrix, aspect='auto', cmap=SEISMIC_CMAP, vmin=vmin, vmax=vmax)
    ax6.set_title('Denoised Data')
    
    # Bottom row: Frequency analysis
    ax7 = fig.add_subplot(gs[2, :])
    
    # Compute and plot average frequency spectra
    fs = 100  # Sampling frequency, Hz
    f, pxx_raw = compute_spectrum(raw_matrix, fs)
    f, pxx_preprocessed = compute_spectrum(preprocessed_matrix, fs)
    f, pxx_denoised = compute_spectrum(denoised_matrix, fs)
    
    ax7.semilogy(f, pxx_raw, 'b-', alpha=0.7, label='Raw')
    ax7.semilogy(f, pxx_preprocessed, 'g-', alpha=0.7, label='Preprocessed')
    ax7.semilogy(f, pxx_denoised, 'r-', alpha=0.7, label='Denoised')
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('Power Spectral Density')
    ax7.set_title('Average Frequency Spectrum')
    ax7.legend()
    ax7.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add colorbar for 2D plots
    cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.3])
    fig.colorbar(im4, cax=cbar_ax, orientation='vertical')
    
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {output_file}")
    
    return fig

def plot_singular_values(singular_values, rank_used=None, output_file=None):
    """
    Plot singular values and their decay.
    
    Parameters
    ----------
    singular_values : numpy.ndarray
        Array of singular values
    rank_used : int, optional
        Rank used for denoising, by default None
    output_file : str, optional
        Path to save figure, by default None
    """
    logger.info("Generating singular values plot")
    
    plt.figure(figsize=(10, 8))
    
    # Create grid for subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Plot singular values
    ax1 = plt.subplot(gs[0])
    ax1.plot(singular_values, 'bo-', markersize=4)
    if rank_used is not None:
        ax1.axvline(x=rank_used-0.5, color='r', linestyle='--', 
                   label=f'Rank Used = {rank_used}')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Singular Values')
    ax1.set_xlim(-0.5, len(singular_values) - 0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot normalized cumulative energy
    ax2 = plt.subplot(gs[1])
    cumulative_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    ax2.plot(cumulative_energy, 'go-', markersize=4)
    if rank_used is not None:
        ax2.axvline(x=rank_used-0.5, color='r', linestyle='--')
        energy_captured = cumulative_energy[rank_used-1]
        ax2.text(rank_used+1, energy_captured-0.05, 
                f'Energy: {energy_captured:.3f}', 
                color='r', fontsize=9)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Cumulative Energy')
    ax2.set_title('Normalized Cumulative Energy')
    ax2.set_xlim(-0.5, len(singular_values) - 0.5)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {output_file}")
        
    return plt.gcf()

def compute_spectrum(data, fs):
    """
    Compute average power spectrum of data.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data matrix (traces x samples)
    fs : float
        Sampling frequency in Hz
    
    Returns
    -------
    f : numpy.ndarray
        Frequency array
    pxx : numpy.ndarray
        Power spectral density
    """
    from scipy import signal
    
    n_traces, n_samples = data.shape
    
    # Average across all traces
    pxx_sum = 0
    for i in range(n_traces):
        f, pxx = signal.welch(data[i], fs, nperseg=min(256, n_samples))
        pxx_sum += pxx
    
    pxx_avg = pxx_sum / n_traces
    return f, pxx_avg

# Example usage
if __name__ == "__main__":
    from src.data.dataset import SeismicDataLoader
    from src.preprocessing.preprocess import preprocess_seismic_data
    from src.models.svd import SVDDenoiser
    
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    loader = SeismicDataLoader("data/raw")
    data = loader.load_data()
    
    # Preprocess
    preprocessed = preprocess_seismic_data(data)
    
    # Denoise
    denoiser = SVDDenoiser(rank=5)
    result = denoiser.denoise(preprocessed)
    
    # Generate plots
    plot_comparison(data, preprocessed, result, "comparison.png")
    plot_singular_values(result['singular_values'], result['rank_used'], "singular_values.png")
    
    plt.show() 