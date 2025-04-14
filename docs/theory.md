# Theory: Low-Rank Matrix Approximation for Seismic Signal Denoising

This document provides a theoretical background on the low-rank matrix approximation techniques used in this project for seismic signal denoising.

## Background: Seismic Signals and Noise

Seismic data typically consists of multiple time series (traces) recorded by sensors (geophones or seismometers) that capture ground motion. These signals contain important geophysical information but are often contaminated by various types of noise:

- **Random noise**: Background vibrations, instrument noise
- **Coherent noise**: Surface waves, multiples, ground roll
- **Impulse noise**: Spikes, outliers

The goal of denoising is to separate the meaningful signal from these noise components without distorting the important features.

## Low-Rank Matrix Approximation Principles

Low-rank matrix approximation is based on the principle that many natural signals, including seismic data, have an intrinsic low-dimensional structure, while noise typically spans a higher-dimensional space. By representing the data as a matrix and finding a lower-rank approximation, we can potentially separate signal from noise.

For seismic data, we organize the traces into a matrix where:
- Each row represents a different trace (sensor recording)
- Each column represents a time sample

## Method 1: Singular Value Decomposition (SVD)

SVD is a fundamental matrix factorization technique that decomposes a matrix \(X\) into three components:

\[ X = U \Sigma V^T \]

Where:
- \(U\) contains the left singular vectors (spatial patterns)
- \(\Sigma\) is a diagonal matrix of singular values (importance of each pattern)
- \(V^T\) contains the right singular vectors (temporal patterns)

For denoising, we truncate this decomposition to keep only the top \(k\) singular values and vectors:

\[ X_{denoised} = U_k \Sigma_k V_k^T \]

This works because the signal typically concentrates in the first few singular vectors with large singular values, while noise tends to be distributed across all components.

## Method 2: Robust Principal Component Analysis (RPCA)

RPCA extends the basic SVD approach by explicitly modeling the data as a sum of a low-rank component \(L\) (the signal) and a sparse component \(S\) (the noise):

\[ X = L + S \]

This is particularly useful for seismic data which may contain outliers or impulse noise. RPCA solves the following optimization problem:

\[ \min_{L,S} \|L\|_* + \lambda\|S\|_1 \quad \text{subject to} \quad X = L + S \]

Where \(\|L\|_*\) is the nuclear norm (sum of singular values) that promotes low-rank solutions, and \(\|S\|_1\) is the L1-norm that promotes sparsity.

The implementation in this project uses the Alternating Direction Method of Multipliers (ADMM) to solve this optimization problem efficiently.

## Method 3: Hankel Matrix SVD (Singular Spectrum Analysis)

Hankel SVD, also known as Singular Spectrum Analysis (SSA), works on a single time series by:

1. **Embedding**: Transforming the 1D time series into a 2D Hankel matrix where each row is a delayed version of the signal:

   \[ H = \begin{bmatrix} 
   x_1 & x_2 & \cdots & x_{N-L+1} \\
   x_2 & x_3 & \cdots & x_{N-L+2} \\
   \vdots & \vdots & \ddots & \vdots \\
   x_L & x_{L+1} & \cdots & x_N
   \end{bmatrix} \]

2. **Decomposition**: Applying SVD to the Hankel matrix

3. **Grouping**: Selecting only the dominant components (based on singular values)

4. **Averaging**: Reconstructing the denoised signal by averaging along the anti-diagonals of the reconstructed Hankel matrix

This method is particularly effective for seismic signals because it exploits the temporal structure and can separate oscillatory components from noise.

## Parameter Selection

Critical parameters for these methods include:

- **Rank (k)**: The number of components to keep in the low-rank approximation. Too low may lose signal details; too high may retain noise.
- **Lambda** (for RPCA): Controls the trade-off between the low-rank and sparse components.
- **Window length** (for Hankel SVD): The embedding dimension, typically chosen based on the dominant periods in the data.

The optimal parameters often depend on the specific characteristics of the seismic data and the noise.

## Performance Metrics

Common metrics to evaluate denoising performance include:

- **Signal-to-Noise Ratio (SNR)**: Measures the ratio of signal power to noise power
- **Mean Squared Error (MSE)**: Measures the average squared difference between the clean and denoised signals
- **Structural Similarity Index (SSIM)**: Measures the perceived similarity between clean and denoised signals

## References

1. Oropeza, V., & Sacchi, M. (2011). Simultaneous seismic data denoising and reconstruction via multichannel singular spectrum analysis. Geophysics, 76(3), V25-V32.
2. Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal component analysis? Journal of the ACM, 58(3), 1-37.
3. Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum Analysis for time series. Springer Science & Business Media.
4. Trickett, S. (2008). F-xy eigenimage noise suppression. Geophysics, 73(6), V29-V34. 