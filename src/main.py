#!/usr/bin/env python
"""
Main script for seismic signal denoising using low-rank matrix approximation.
"""
import argparse
import logging
import os
from datetime import datetime

from data.dataset import SeismicDataLoader
from models.svd import SVDDenoiser
from models.rpca import RPCADenoiser
from models.hankel import HankelDenoiser
from preprocessing.preprocess import preprocess_seismic_data
from visualization.visualize import plot_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Seismic Signal Denoising')
    parser.add_argument('--data_path', type=str, default='data/raw',
                        help='Path to raw data')
    parser.add_argument('--output_path', type=str, default='data/processed',
                        help='Path to save processed data')
    parser.add_argument('--method', type=str, default='svd',
                        choices=['svd', 'rpca', 'hankel'],
                        help='Denoising method')
    parser.add_argument('--rank', type=int, default=10,
                        help='Rank for low-rank approximation')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    return parser.parse_args()

def main():
    """Run the denoising pipeline."""
    args = parse_args()
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_path, f"{args.method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting denoising with method: {args.method}")
    logger.info(f"Loading data from {args.data_path}")
    
    # Load data
    data_loader = SeismicDataLoader(args.data_path)
    raw_data = data_loader.load_data()
    
    # Preprocess data
    logger.info("Preprocessing data")
    preprocessed_data = preprocess_seismic_data(raw_data)
    
    # Apply denoising
    logger.info(f"Applying {args.method} denoising with rank {args.rank}")
    if args.method == 'svd':
        denoiser = SVDDenoiser(rank=args.rank)
    elif args.method == 'rpca':
        denoiser = RPCADenoiser(rank=args.rank)
    elif args.method == 'hankel':
        denoiser = HankelDenoiser(rank=args.rank)
    else:
        raise ValueError(f"Method {args.method} not implemented yet")
    
    denoised_data = denoiser.denoise(preprocessed_data)
    
    # Save results
    output_file = os.path.join(output_dir, "denoised_data.npz")
    logger.info(f"Saving results to {output_file}")
    data_loader.save_data(denoised_data, output_file)
    
    # Visualize if requested
    if args.visualize:
        logger.info("Generating visualizations")
        figure_path = os.path.join(output_dir, "comparison.png")
        plot_comparison(raw_data, preprocessed_data, denoised_data, figure_path)
        
        # Save singular values plot if available
        if 'singular_values' in denoised_data:
            from visualization.visualize import plot_singular_values
            sv_figure_path = os.path.join(output_dir, "singular_values.png")
            plot_singular_values(
                denoised_data['singular_values'], 
                denoised_data.get('rank_used'), 
                sv_figure_path
            )
    
    logger.info("Denoising completed successfully")

if __name__ == "__main__":
    main() 