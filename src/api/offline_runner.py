#!/usr/bin/env python3
"""
RC2 Offline Inference Runner for RSNA Intracranial Aneurysm Detection

This script scans a directory of DICOM series, calls a lightweight predictor adapter,
and writes a Kaggle-ready submission.csv. It is fully offline, deterministic, and
CPU-friendly. It prefers a compact 3D model if available and safely falls back to
a deterministic stub that returns calibrated priors.

Usage:
  python -m src.api.offline_runner --series-root /path/to/series --output submission.csv
  # Each subdirectory of --series-root is expected to correspond to a SeriesInstanceUID
  # and contain that series' DICOM files.
"""
import os
import sys
import gc
import json
import time
import math
import logging
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Union

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import local modules
from src.utils.path_utils import PATHS

# Import project modules
from src.utils.predictor_loader import load_predictor_adapter, setup_environment
from src.constants import (
    RSNA_ALL_LABELS,
    RSNA_LOCATION_LABELS,
    RSNA_ANEURYSM_PRESENT_LABEL,
    REQUIRED_COLUMNS,
    DEFAULT_SUBMISSION_CSV
)
from src.utils.calibration import load_calibrator
from src.validate_submission import validate_submission_file

# Try to import optional dependencies
try:
    import polars as pl
except ImportError:
    pl = None
    
try:
    import pydicom
except ImportError:
    pydicom = None

import numpy as np
import pandas as pd

# Set up environment and logging
setup_environment()
logger = logging.getLogger(__name__)

# Constants
ID_COL = "SeriesInstanceUID"

# Type aliases
PredictionResult = Dict[str, float]




def _is_dicom_file(path: Path) -> bool:
    """Check if a file is likely a DICOM file."""
    if not path.is_file():
        return False
    
    # Check common DICOM extensions
    if path.suffix.lower() in ('.dcm', '.dicom', '.ima'):
        return True
    
    # Check for DICOM magic number
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False


def _iter_series_dirs(series_root: Path) -> List[Path]:
    """Iterate over valid DICOM series directories."""
    if not series_root.exists() or not series_root.is_dir():
        raise ValueError(f"Series root not found or not a directory: {series_root}")
    
    # Get all subdirectories that contain DICOM files
    series_dirs = []
    for entry in series_root.iterdir():
        if not entry.is_dir():
            continue
            
        # Check if directory contains DICOM files
        has_dicoms = any(_is_dicom_file(p) for p in entry.glob('*'))
        if has_dicoms:
            series_dirs.append(entry)
    
    if not series_dirs:
        raise ValueError(f"No DICOM series found in {series_root}")
    
    # Sort for deterministic order
    return sorted(series_dirs, key=lambda p: p.name)


def validate_frame(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate a prediction DataFrame against the RSNA schema."""
    errors = []
    
    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(sorted(missing_cols))}")
    
    # Check for unexpected columns
    extra_cols = set(df.columns) - set(REQUIRED_COLUMNS)
    if extra_cols:
        errors.append(f"Unexpected columns: {', '.join(sorted(extra_cols))}")
    
    # Check for null values
    for col in RSNA_ALL_LABELS:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Null values in column '{col}': {null_count}")
    
    # Check value ranges
    for col in RSNA_ALL_LABELS:
        if col in df.columns and not df[col].isnull().all():
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < 0 or max_val > 1:
                errors.append(
                    f"Out-of-range values in column '{col}': min={min_val}, max={max_val}"
                )
    
    return len(errors) == 0, errors


def run_offline_inference(
    series_root: Union[str, Path],
    output_path: Union[str, Path] = None,
    limit: Optional[int] = None,
    calibrate: bool = True
) -> pd.DataFrame:
    """Run offline inference on DICOM series.
    
    Args:
        series_root: Path to directory containing DICOM series subdirectories
        output_path: Path to write submission CSV (default: submission.csv in current dir)
        limit: Maximum number of series to process (for testing)
        calibrate: Whether to apply calibration if available
        
    Returns:
        DataFrame containing the submission results
    """
    start_time = time.time()
    series_root = Path(series_root).resolve()
    output_path = Path(output_path) if output_path else Path(DEFAULT_SUBMISSION_CSV)
    
    logger.info(f"Starting offline inference with series root: {series_root}")
    
    # Load predictor
    predict_fn, predictor_name = _load_compact3d_predictor()
    logger.info(f"Using predictor: {predictor_name}")
    
    # Load calibrator if available
    calibrator = load_calibrator() if calibrate else None
    if calibrator:
        logger.info("Using calibration")
    
    # Get series directories
    try:
        series_dirs = _iter_series_dirs(series_root)
        if limit:
            series_dirs = series_dirs[:limit]
            logger.info(f"Limiting to first {limit} series")
    except Exception as e:
        logger.error(f"Error finding series directories: {e}")
        raise
    
    # Process each series
    results = []
    latencies = []
    failed = []
    
    for i, series_dir in enumerate(series_dirs, 1):
        series_id = series_dir.name
        logger.info(f"Processing [{i}/{len(series_dirs)}] {series_id}")
        iter_start = time.time()
        
        try:
            # Get predictions
            preds = predict_fn(str(series_dir))
            
            # Apply calibration if available
            if calibrator:
                preds = calibrator.calibrate_predictions(preds)
            
            # Create result row
            row = {"SeriesInstanceUID": series_id, **preds}
            results.append(row)
            
            # Log success
            iter_time = time.time() - iter_start
            latencies.append(iter_time)
            logger.debug(f"Completed {series_id} in {iter_time:.2f}s")
            
        except Exception as e:
            iter_time = time.time() - iter_start
            logger.error(f"Failed to process {series_id} after {iter_time:.2f}s: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            failed.append(series_id)
    
    # Check if we have any results
    if not results:
        raise RuntimeError("No predictions generated successfully")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure all required columns are present
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col == ID_COL:
                raise ValueError("Missing required column: SeriesInstanceUID")
            df[col] = 0.0  # Fill missing prediction columns with zeros
    
    # Reorder columns
    df = df[REQUIRED_COLUMNS]
    
    # Validate the output
    is_valid, errors = validate_frame(df)
    if not is_valid:
        for error in errors:
            logger.error(f"Validation error: {error}")
        if len(errors) > 5:
            logger.error(f"... and {len(errors) - 5} more errors")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use polars if available (faster for large datasets)
    if pl is not None:
        try:
            pl_df = pl.from_pandas(df)
            pl_df.write_csv(output_path)
        except Exception as e:
            logger.warning(f"Failed to write with polars, falling back to pandas: {e}")
            df.to_csv(output_path, index=False, float_format="%.6f")
    else:
        df.to_csv(output_path, index=False, float_format="%.6f")
    
    # Log summary
    total_time = time.time() - start_time
    avg_latency = np.mean(latencies) if latencies else 0
    
    summary = [
        "=" * 70,
        f"INFERENCE SUMMARY ({predictor_name})",
        "=" * 70,
        f"Processed: {len(results)}/{len(series_dirs)} series",
        f"Time: {total_time:.1f}s total, {avg_latency:.2f}s avg per series",
        f"Output: {output_path}",
    ]
    
    if failed:
        summary.extend([
            f"\nWARNING: Failed to process {len(failed)} series:",
            *[f"  {i+1}. {sid}" for i, sid in enumerate(failed[:10])],
            f"  ... and {len(failed) - 10} more" if len(failed) > 10 else ""
        ])
    
    logger.info("\n".join(filter(None, summary)))
    
    return df


def main(series_root: str = None, output: str = None, limit: int = None, 
         calibrate: bool = True, verbose: bool = False, **kwargs):
    """
    Run the offline inference pipeline.
    
    Args:
        series_root: Path to directory containing DICOM series subdirectories
        output: Output CSV path (default: submission.csv in current directory)
        limit: Maximum number of series to process (for testing)
        calibrate: Whether to apply calibration if available
        verbose: Enable verbose logging
        **kwargs: Additional keyword arguments (ignored)
    """
    # Handle command line arguments if not provided
    if series_root is None:
        parser = argparse.ArgumentParser(description='RSNA Aneurysm Detection - Offline Inference Runner')
        parser.add_argument('--series-root', type=str, required=True,
                          help='Path to directory containing DICOM series subdirectories')
        parser.add_argument('--output', type=str, default=DEFAULT_SUBMISSION_CSV,
                          help=f'Output CSV path (default: {DEFAULT_SUBMISSION_CSV})')
        parser.add_argument('--limit', type=int, default=None,
                          help='Limit number of series to process (for testing)')
        parser.add_argument('--no-calibration', action='store_false', dest='calibrate',
                          help='Disable calibration')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose logging')
        
        args = parser.parse_args()
        series_root = args.series_root
        output = args.output
        limit = args.limit
        calibrate = args.calibrate
        verbose = args.verbose
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run inference
    try:
        run_offline_inference(
            series_root=series_root,
            output_path=output,
            limit=limit,
            calibrate=calibrate
        )
        logger.info(f"Successfully wrote submission to {output}")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        if verbose:
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
