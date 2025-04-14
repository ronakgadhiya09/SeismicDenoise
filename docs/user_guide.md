# User Guide: Signal Denoising in Seismology

This guide provides instructions on how to use the seismic signal denoising toolkit.

## Installation

1. Clone the repository
2. Navigate to the project directory
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

The main script (`src/main.py`) provides a command-line interface for denoising seismic signals. Here's how to use it:

```bash
python src/main.py --data_path data/raw --method svd --rank 10 --visualize
```

### Command-Line Arguments

- `--data_path`: Path to the directory containing raw seismic data files (default: `data/raw`)
- `--output_path`: Path where processed data will be saved (default: `data/processed`)
- `--method`: Denoising method to use, options are `svd`, `rpca`, or `hankel` (default: `svd`)
- `--rank`: Rank for low-rank approximation (default: `10`)
- `--visualize`: Flag to generate visualization plots (optional)

## Data Format

The toolkit supports seismic data in various formats:

- **MiniSEED (.mseed)** files (standard seismic data format)
- **SAC** files
- **NumPy (.npy, .npz)** files containing seismic traces

If no data files are found, the system will generate synthetic data for demonstration purposes.

## Denoising Methods

### 1. SVD Denoising

Singular Value Decomposition (SVD) denoising works by:
1. Decomposing the multichannel seismic data matrix into U, Σ, and V components
2. Truncating to keep only the top k singular values and vectors
3. Reconstructing the denoised matrix from the truncated components

Example:
```bash
python src/main.py --method svd --rank 5
```

### 2. Robust PCA Denoising

Robust Principal Component Analysis (RPCA) extends SVD by:
1. Decomposing the data matrix into a low-rank component (signal) and a sparse component (noise)
2. Using an iterative optimization algorithm (ADMM) to separate these components

Example:
```bash
python src/main.py --method rpca --rank 5
```

### 3. Hankel Matrix SVD Denoising

Hankel SVD (Singular Spectrum Analysis) works by:
1. Embedding each seismic trace into a Hankel matrix
2. Performing SVD on this matrix
3. Truncating to keep only the dominant components
4. Reconstructing the denoised signal by averaging anti-diagonals

Example:
```bash
python src/main.py --method hankel --rank 5
```

## Jupyter Notebooks

For interactive exploration and visualization, Jupyter notebooks are provided in the `notebooks/` directory:

- `Seismic_Denoising_Demo.ipynb`: Comprehensive demonstration of all denoising methods

To run the notebooks:
```bash
jupyter notebook notebooks/
```

## Example Workflow

1. **Prepare your data**: Place your seismic files in the `data/raw/` directory
2. **Explore the data**: Use Jupyter notebooks to visualize and understand your data
3. **Run denoising**: Apply different methods with varying parameters
4. **Compare results**: Use the visualization tools to evaluate method performance
5. **Save processed data**: The denoised data is automatically saved to `data/processed/`

## Customization

You can extend the toolkit by:

- Implementing new denoising methods in `src/models/`
- Adding custom preprocessing steps in `src/preprocessing/preprocess.py`
- Creating new visualization functions in `src/visualization/visualize.py` 