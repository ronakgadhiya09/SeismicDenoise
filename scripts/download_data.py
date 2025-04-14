#!/usr/bin/env python
"""
Download example seismic data for the project.
"""
import os
import logging
import argparse
import urllib.request
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available datasets with URLs
DATASETS = {
    'synthetic': {
        'url': 'https://github.com/earthinversion/SeismicTraining/raw/main/data/synthetic_seismic.zip',
        'description': 'Synthetic seismic dataset with clean and noisy versions'
    },
    'earthquake': {
        'url': 'https://github.com/obspy/obspy/raw/master/obspy/io/mseed/tests/data/test.mseed',
        'description': 'Small earthquake recording in MiniSEED format'
    }
}

def download_file(url, output_path):
    """
    Download a file from a URL to the specified path.
    
    Parameters
    ----------
    url : str
        URL to download from
    output_path : str
        Local path to save the file
    """
    logger.info(f"Downloading from {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Downloaded to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return False

def extract_zip(zip_path, extract_dir):
    """
    Extract a ZIP file.
    
    Parameters
    ----------
    zip_path : str
        Path to the ZIP file
    extract_dir : str
        Directory to extract to
    """
    logger.info(f"Extracting {zip_path} to {extract_dir}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info("Extraction complete")
        return True
    except Exception as e:
        logger.error(f"Failed to extract: {e}")
        return False

def download_dataset(dataset_name, output_dir):
    """
    Download a specific dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to download
    output_dir : str
        Directory to save the dataset
    """
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False
    
    dataset = DATASETS[dataset_name]
    url = dataset['url']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine file name and path
    file_name = url.split('/')[-1]
    output_path = os.path.join(output_dir, file_name)
    
    # Download the file
    success = download_file(url, output_path)
    if not success:
        return False
    
    # Extract if it's a ZIP file
    if file_name.endswith('.zip'):
        extract_success = extract_zip(output_path, output_dir)
        # Remove the ZIP file after extraction
        if extract_success:
            os.remove(output_path)
    
    return True

def main():
    """Main function to download datasets."""
    parser = argparse.ArgumentParser(description='Download seismic datasets')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=list(DATASETS.keys()) + ['all'],
                        help='Dataset to download')
    parser.add_argument('--output', type=str, default='data/raw',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Display available datasets
    logger.info("Available datasets:")
    for name, details in DATASETS.items():
        logger.info(f"  - {name}: {details['description']}")
    
    # Download selected dataset(s)
    if args.dataset == 'all':
        logger.info("Downloading all datasets")
        for name in DATASETS:
            download_dataset(name, args.output)
    else:
        logger.info(f"Downloading dataset: {args.dataset}")
        download_dataset(args.dataset, args.output)
    
    logger.info("Download process completed")

if __name__ == "__main__":
    main() 