# 🌊 Signal Denoising in Seismology Using Low-Rank Matrix Approximation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ObsPy](https://img.shields.io/badge/ObsPy-2.0.0-orange)](https://github.com/obspy/obspy)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-lightblue)](https://numpy.org/)

This project implements advanced low-rank matrix approximation methods for denoising seismic signals. It demonstrates how matrix decomposition techniques can effectively separate signal from noise in seismological data by exploiting the inherent low-rank structure of seismic matrices.

<p align="center">
  <img src="docs/Seismic Denoising Example.jpg" alt="Seismic Denoising Example" width="700"/>
</p>

## 📋 Features

- **Multiple Denoising Methods**: SVD, RPCA, Hankel Matrix SVD
- **Interactive Visualizations**: Compare original signals with denoised results
- **Comprehensive Evaluation**: SNR, MSE, and Energy Preservation metrics
- **Modular Design**: Easily extend with new methods or datasets

## 🗂️ Project Structure

```
.
├── data/
│   ├── raw/                # Raw seismic datasets
│   └── processed/          # Cleaned and preprocessed data
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks for exploration and visualization
├── src/                    # Source code
│   ├── data/               # Data loading and management
│   ├── models/             # Denoising algorithms
│   ├── preprocessing/      # Signal preprocessing
│   ├── utils/              # Utility functions
│   └── visualization/      # Plotting and visualization
└── tests/                  # Unit tests
```

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ronakgadhiya09/SeismicDenoise.git
   cd SeismicDenoise
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running Denoising Methods

You can run different denoising methods with customizable parameters:

```bash
# SVD denoising
python src/main.py --method svd --rank 5 --visualize

# RPCA denoising
python src/main.py --method rpca --rank 5 --visualize

# Hankel SVD denoising
python src/main.py --method hankel --rank 5 --window 50 --visualize
```

### Visualization Examples

For detailed analysis and method comparison, explore the Jupyter notebooks in the `notebooks/` directory:

```bash
jupyter notebook notebooks/method_comparison.ipynb
```

## 📊 Denoising Methods

### Singular Value Decomposition (SVD)

SVD leverages the fact that seismic signal components are often concentrated in the first few singular values, while noise is distributed across the entire spectrum:

```python
# Basic usage
from src.models.svd import SVDDenoiser

denoiser = SVDDenoiser(rank=5)
denoised_signal = denoiser.denoise(noisy_signal)
```

### Robust Principal Component Analysis (RPCA)

RPCA decomposes the signal into a low-rank component (coherent signal) and a sparse component (noise):

```python
# Basic usage
from src.models.rpca import RPCADenoiser

denoiser = RPCADenoiser(rank=5, lambda_param=0.1)
denoised_signal = denoiser.denoise(noisy_signal)
```

### Hankel Matrix SVD

Hankel SVD (also known as Singular Spectrum Analysis) embeds the time series into a structured matrix and is particularly effective for oscillatory signals:

```python
# Basic usage
from src.models.hankel import HankelDenoiser

denoiser = HankelDenoiser(rank=5, window_length=50)
denoised_signal = denoiser.denoise(noisy_signal)
```

## 📈 Performance Comparison

| Method | SNR Improvement | MSE Reduction | Energy Preservation |
|--------|----------------|---------------|---------------------|
| SVD    | 6.8-9.3 dB     | 75-85%        | 87-94%              |
| RPCA   | 7.7-10.1 dB    | 80-90%        | 89-96%              |
| Hankel | 6.4-11.8 dB    | 70-88%        | 91-93%              |

## 📦 Datasets

This project uses seismic data from [IRIS Seismology Data Service](https://www.iris.edu/hq/) which provides open access to earthquake waveform data.

Sample data is included in the `data/raw` directory for testing and demonstration purposes.

## 📚 References

- Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust Principal Component Analysis? Journal of the ACM, 58(3), 1-37.
- Oropeza, V., & Sacchi, M. (2011). Simultaneous seismic data denoising and reconstruction via multichannel singular spectrum analysis. Geophysics, 76(3), V25-V32.
- Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum Analysis for Time Series. Springer Science & Business Media.
- Trickett, S. (2008). F-xy Eigenimage Noise Suppression. Geophysics, 73(6), V29-V34.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

- Ronak Gadhiya ([@ronakgadhiya09](https://github.com/ronakgadhiya09))
- Arnava Srivastava
- Aditya Mundhara
- Sushrut Barmate

## 📬 Contact

For questions and feedback, please open an issue on GitHub or contact us at [yourname@example.com](mailto:b22ai052@iitj.ac.in). 
