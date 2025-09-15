#!/usr/bin/env python3
"""
Generate isotonic calibration points from OOF predictions.

Usage:
    python calibrate_isotonic.py oof.csv --output-dir ./calib --samples 100

Outputs one JSON file per label with calibration points in the format:
{
  "x": [0.0, 0.1, ..., 1.0],  # Model outputs (binned)
  "y": [0.0, 0.15, ..., 1.0]   # Calibrated probabilities
}
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from tqdm import tqdm

# RSNA labels (excluding 'Aneurysm Present' which is handled separately)
RSNA_LOCATIONS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation'
]

def load_oof(oof_path: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Load OOF predictions and validate required columns."""
    df = pd.read_csv(oof_path)
    
    # Check required columns
    required_cols = ['SeriesInstanceUID', 'StudyInstanceUID', 'PatientID'] + RSNA_LOCATIONS + [f'pred_{loc}' for loc in RSNA_LOCATIONS]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in OOF file: {missing}")
    
    # Extract predictions and targets
    y_true = {loc: df[loc].values for loc in RSNA_LOCATIONS}
    y_pred = {loc: df[f'pred_{loc}'].values for loc in RSNA_LOCATIONS}
    
    return df, y_true, y_pred

def calibrate_isotonic(y_true: np.ndarray, y_pred: np.ndarray, n_samples: int = 100) -> Dict:
    """Fit isotonic regression and return calibration points."""
    # Skip if no positive samples
    if np.sum(y_true) == 0:
        return {
            'x': [0.0, 1.0],
            'y': [0.0, 1.0],
            'brier': float('nan'),
            'n_pos': 0
        }
    
    # Fit isotonic regression
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(y_pred, y_true)
    
    # Generate calibration curve points
    x = np.linspace(0, 1, n_samples)
    y = ir.predict(x)
    
    # Calculate Brier score before calibration
    brier = brier_score_loss(y_true, y_pred)
    
    return {
        'x': x.tolist(),
        'y': y.tolist(),
        'brier': float(brier),
        'n_pos': int(np.sum(y_true))
    }

def main():
    parser = argparse.ArgumentParser(description='Generate isotonic calibration points from OOF predictions')
    parser.add_argument('oof_path', type=str, help='Path to OOF predictions CSV')
    parser.add_argument('--output-dir', type=str, default='./calib', 
                       help='Output directory for calibration JSON files')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of points to sample for calibration curve')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load OOF data
    print(f"Loading OOF predictions from {args.oof_path}...")
    df, y_true, y_pred = load_oof(args.oof_path)
    
    # Calibrate each location
    print(f"Fitting isotonic regression for {len(RSNA_LOCATIONS)} locations...")
    results = {}
    
    for loc in tqdm(RSNA_LOCATIONS):
        calib = calibrate_isotonic(
            y_true[loc], 
            y_pred[loc],
            n_samples=args.samples
        )
        results[loc] = calib
        
        # Save individual calibration files
        output_path = output_dir / f"calib_{loc.lower().replace(' ', '_')}.json"
        with open(output_path, 'w') as f:
            json.dump(calib, f, indent=2)
    
    # Save summary
    summary = {
        'locations': RSNA_LOCATIONS,
        'brier_scores': {loc: results[loc]['brier'] for loc in RSNA_LOCATIONS},
        'n_positives': {loc: results[loc]['n_pos'] for loc in RSNA_LOCATIONS},
        'total_samples': len(df),
        'calibration_points': args.samples
    }
    
    with open(output_dir / 'calibration_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\nCalibration Summary:")
    print(f"{'Location':<45} {'Brier Score':<12} {'Positives':<10}")
    print("-" * 70)
    for loc in RSNA_LOCATIONS:
        print(f"{loc:<45} {results[loc]['brier']:.6f}  {results[loc]['n_pos']:>5}")
    
    print(f"\nCalibration files saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
