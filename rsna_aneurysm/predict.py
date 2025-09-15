#!/usr/bin/env python3
"""Command-line interface for RSNA Aneurysm Detection."""
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from rsna_aneurysm import AneurysmPredictor, LABELS


def find_series_dirs(root_dir: str) -> List[str]:
    """Find directories containing DICOM series.
    
    Args:
        root_dir: Root directory containing subdirectories with DICOM files.
        
    Returns:
        List of paths to directories containing DICOM series.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"Directory not found: {root_dir}")
    
    series_dirs = []
    for entry in root_path.iterdir():
        if entry.is_dir():
            # Check if directory contains DICOM files
            has_dicom = any(
                f.suffix.lower() in ('.dcm', '.dicom', '') 
                for f in entry.glob('*')
            )
            if has_dicom:
                series_dirs.append(str(entry))
    
    return series_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Run RSNA Aneurysm Detection on DICOM series."
    )
    parser.add_argument(
        "--series-root",
        required=True,
        help="Root directory containing subdirectories of DICOM series"
    )
    parser.add_argument(
        "--out",
        default="predictions.csv",
        help="Output CSV file path (default: predictions.csv)"
    )
    parser.add_argument(
        "--weights",
        help="Path to model weights file (default: use random weights)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: auto-detect)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of series to process (0 for no limit)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    args = parser.parse_args()
    
    # Find DICOM series directories
    print(f"Scanning {args.series_root} for DICOM series...")
    series_dirs = find_series_dirs(args.series_root)
    
    if not series_dirs:
        print("No DICOM series found in the specified directory.")
        return
    
    # Apply limit if specified
    if args.limit > 0:
        series_dirs = series_dirs[:args.limit]
    
    print(f"Found {len(series_dirs)} DICOM series to process.")
    
    # Initialize predictor
    predictor = AneurysmPredictor(weights_path=args.weights, device=args.device)
    
    # Process each series
    results = []
    for series_dir in tqdm(series_dirs, desc="Processing series"):
        try:
            # Get predictions as a dictionary
            pred_dict = predictor.predict_series(series_dir)
            pred_dict["SeriesInstanceUID"] = os.path.basename(series_dir.rstrip('/'))
            results.append(pred_dict)
        except Exception as e:
            print(f"Error processing {series_dir}: {e}")
    
    if not results:
        print("No predictions were generated.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure all label columns are present
    for label in LABELS:
        if label not in df.columns:
            df[label] = 0.0
    
    # Reorder columns to put SeriesInstanceUID first, then labels
    columns = ["SeriesInstanceUID"] + LABELS
    df = df[columns]
    
    # Save results
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "csv":
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:  # json
        # Convert to list of records for better JSON serialization
        records = df.to_dict(orient='records')
        with open(output_path, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
