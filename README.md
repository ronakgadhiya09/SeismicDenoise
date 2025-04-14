# Signal Denoising in Seismology Using Low-Rank Matrix Approximation

This project implements various low-rank matrix approximation methods for denoising seismic signals. It demonstrates how matrix decomposition techniques can effectively separate signal from noise in seismological data.

## Project Structure

```
.
├── data/
│   ├── raw/                # Raw seismic datasets
│   └── processed/          # Cleaned and preprocessed data
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks for exploration and visualization
└── src/                    # Source code
    ├── data/               # Data loading and management
    ├── models/             # Denoising algorithms
    ├── preprocessing/      # Signal preprocessing
    ├── utils/              # Utility functions
    └── visualization/      # Plotting and visualization
```

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to denoise seismic signals:

```
python src/main.py
```

For detailed analysis and method comparison, explore the Jupyter notebooks in the `notebooks/` directory.

## Datasets

This project uses seismic data from [IRIS Seismology Data Service](https://www.iris.edu/hq/) which provides open access to earthquake waveform data.

## Methods

The following low-rank matrix approximation techniques are implemented:
- Singular Value Decomposition (SVD)
- Robust Principal Component Analysis (RPCA)
- Matrix Completion
- Hankel Matrix SVD

## References

- Oropeza, V., & Sacchi, M. (2011). Simultaneous seismic data denoising and reconstruction via multichannel singular spectrum analysis. Geophysics, 76(3), V25-V32.
- Basterrech, S. (2021). A Survey of Low-Rank Matrix Factorization Methods for Recommender Systems. Applied Sciences, 11(20), 9483. 