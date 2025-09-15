#!/usr/bin/env python3
"""
Main entry point for the RSNA Aneurysm Detection Pipeline.

This script ensures proper Python path setup before importing and running other modules.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    # Import here to ensure path is set up first
    from src.api.offline_runner import main as offline_runner_main
    from src.train_model import main as train_model_main
    
    import argparse
    
    parser = argparse.ArgumentParser(description='RSNA Aneurysm Detection Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-dir', type=str, help='Path to data directory')
    train_parser.add_argument('--output-dir', type=str, help='Output directory for models')
    train_parser.add_argument('--config', type=str, help='Path to config file')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--series-root', type=str, required=True, 
                            help='Path to directory containing DICOM series')
    infer_parser.add_argument('--output', type=str, required=True,
                            help='Path to output CSV file')
    infer_parser.add_argument('--limit', type=int, help='Maximum number of series to process')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Convert args to a dictionary and remove the command
        train_args = vars(args)
        train_args.pop('command', None)
        train_model_main(**train_args)
    elif args.command == 'infer':
        offline_runner_main()
    else:
        parser.print_help()
        sys.exit(1)
