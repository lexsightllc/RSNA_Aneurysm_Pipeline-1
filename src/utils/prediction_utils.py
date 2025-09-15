"""Utility functions for RSNA Aneurysm Detection predictions."""
from typing import Dict, Any, List, Optional, Union, Tuple
import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(message: str, log_path: Optional[str] = None) -> None:
    """Log message to both console and file if log_path is provided."""
    logger.info(message)
    if log_path:
        with open(log_path, 'a') as f:
            f.write(f"{message}\n")

# RSNA label definitions
RSNA_ALL_LABELS = [
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
    'Other Posterior Circulation',
    'Aneurysm Present'
]

RSNA_LOCATION_LABELS = RSNA_ALL_LABELS[:-1]  # All except 'Aneurysm Present'
RSNA_ANEURYSM_PRESENT_LABEL = 'Aneurysm Present'

# Label synonyms for canonicalization
LABEL_SYNONYMS = {
    'ACom': 'Anterior Communicating Artery',
    'L MCA': 'Left Middle Cerebral Artery',
    'R MCA': 'Right Middle Cerebral Artery',
    'L ACA': 'Left Anterior Cerebral Artery',
    'R ACA': 'Right Anterior Cerebral Artery',
    'L PCom': 'Left Posterior Communicating Artery',
    'R PCom': 'Right Posterior Communicating Artery',
    'Basilar': 'Basilar Tip',
    'Aneurysm': 'Aneurysm Present',
    'AneurysmPresent': 'Aneurysm Present',
    'aneurysm_present': 'Aneurysm Present',
    'series_uid': 'SeriesInstanceUID',
    'study_uid': 'StudyInstanceUID'
}

# Create canonical label mappings
CANONICAL_LABELS = {lbl.lower(): lbl for lbl in RSNA_ALL_LABELS}
CANONICAL_LABELS.update({k.lower(): v for k, v in LABEL_SYNONYMS.items()})

# Add common variations for location labels
for loc in ['Left', 'Right', 'L ', 'R ']:
    for part in ['ICA', 'MCA', 'ACA', 'PCom']:
        short = f"{loc}{part}"
        long = f"{loc} {part}" if ' ' in loc else f"{loc} {part}"
        CANONICAL_LABELS[short.lower()] = long

# Add common abbreviations for posterior circulation
CANONICAL_LABELS.update({
    'pcom': 'Posterior Communicating Artery',
    'basilar': 'Basilar Tip',
    'pca': 'Posterior Cerebral Artery',
    'pcomm': 'Posterior Communicating Artery',
    'acom': 'Anterior Communicating Artery',
    'mca': 'Middle Cerebral Artery',
    'aca': 'Anterior Cerebral Artery',
    'ica': 'Internal Carotid Artery'
})

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map predictor/CLI output columns to canonical RSNA labels using synonyms and case-insensitive keys."""
    rename_map = {}
    for col in list(df.columns):
        key = str(col).strip().lower()
        if key in CANONICAL_LABELS:
            target = CANONICAL_LABELS[key]
            if target != col:
                rename_map[col] = target
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def reconcile_presence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'Aneurysm Present' is at least a fraction of the max location probability per row.
    This maintains logical consistency between presence and location predictions.
    """
    if RSNA_ANEURYSM_PRESENT_LABEL not in df.columns:
        return df
    
    loc_cols = [c for c in RSNA_LOCATION_LABELS if c in df.columns]
    if not loc_cols:
        return df
    
    # Calculate max location probability per row
    loc_max = df[loc_cols].max(axis=1)
    
    # Update presence probability to be at least 95% of max location probability
    # while respecting the [0,1] range
    df[RSNA_ANEURYSM_PRESENT_LABEL] = np.clip(
        np.maximum(
            df[RSNA_ANEURYSM_PRESENT_LABEL].values, 
            (loc_max * 0.95).values
        ),
        1e-6,  # Small epsilon to avoid zeros
        1 - 1e-6  # Just under 1.0 to avoid perfect predictions
    )
    return df

def extract_dicom_metadata(series_dir: str) -> Dict[str, Any]:
    """
    Extract metadata from a sample of DICOM files for reliable slice thickness and phase detection.
    
    Args:
        series_dir: Path to directory containing DICOM files
        
    Returns:
        Dictionary containing extracted metadata including:
        - series_description: Most common series description
        - modality: Most common modality
        - slice_thickness_mm: Median slice thickness in mm
        - is_arterial: Whether this appears to be an arterial phase
        - is_venous: Whether this appears to be a venous phase
        - contrast: Contrast agent information if available
    """
    meta = {
        'series_description': '',
        'modality': '',
        'slice_thickness_mm': None,
        'is_arterial': False,
        'is_venous': False,
        'contrast': ''
    }
    
    try:
        # Sample up to 7 DICOM files to avoid being misled by scouts or corrupt headers
        dcm_paths = []
        for root, _, files in os.walk(series_dir):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dcm_paths.append(os.path.join(root, f))
                    if len(dcm_paths) >= 7:
                        break
            if dcm_paths:  # Found some DICOMs, no need to keep looking
                break
                
        if not dcm_paths:
            logger.warning(f"No DICOM files found in {series_dir}")
            return meta

        # Collect metadata from sampled DICOMs
        descs, modalities, thicknesses, agents = [], [], [], []
        
        for p in dcm_paths:
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                
                # Extract relevant fields
                if hasattr(ds, 'SeriesDescription'):
                    descs.append(str(ds.SeriesDescription))
                    
                if hasattr(ds, 'Modality'):
                    modalities.append(str(ds.Modality))
                    
                if hasattr(ds, 'SliceThickness'):
                    try:
                        thicknesses.append(float(ds.SliceThickness))
                    except (ValueError, TypeError):
                        pass
                        
                if hasattr(ds, 'ContrastBolusAgent'):
                    agents.append(str(ds.ContrastBolusAgent))
                    
            except Exception as e:
                logger.warning(f"Error reading DICOM {p}: {e}")
                continue

        # Determine most common values
        if descs:
            series_desc = max(set(descs), key=descs.count).lower()
            meta['series_description'] = series_desc
            meta['is_arterial'] = any(x in series_desc for x in ['arterial', 'cta', 'angio'])
            meta['is_venous'] = any(x in series_desc for x in ['venous', 'delay'])
            
        if modalities:
            meta['modality'] = max(set(modalities), key=modalities.count)
            
        if thicknesses:
            meta['slice_thickness_mm'] = float(np.median(thicknesses))
            
        if agents:
            meta['contrast'] = max(set(agents), key=agents.count)
            
    except Exception as e:
        logger.error(f"Error extracting DICOM metadata: {e}", exc_info=True)
        
    return meta

def apply_heuristics(predictions: Dict[str, float], metadata: Dict[str, Any]) -> Dict[str, float]:
    """
    Apply heuristics to adjust predictions based on DICOM metadata.
    
    Args:
        predictions: Dictionary of prediction probabilities
        metadata: Dictionary of DICOM metadata
        
    Returns:
        Adjusted predictions
    """
    adjusted = predictions.copy()
    
    # Example phase-based adjustments
    if metadata.get('is_arterial', False):
        # Slightly boost presence in arterial phase
        adjusted[RSNA_ANEURYSM_PRESENT_LABEL] = np.clip(
            adjusted.get(RSNA_ANEURYSM_PRESENT_LABEL, 0) * 1.1,
            1e-6, 1-1e-6
        )
    
    # Example slice thickness adjustments
    slice_thickness = metadata.get('slice_thickness_mm')
    if slice_thickness is not None:
        if slice_thickness <= 1.2:  # Thin slices - boost confidence
            for k in RSNA_LOCATION_LABELS + [RSNA_ANEURYSM_PRESENT_LABEL]:
                if k in adjusted:
                    adjusted[k] = np.clip(adjusted[k] * 1.05, 1e-6, 1-1e-6)
        elif slice_thickness > 1.5:  # Thick slices - reduce confidence
            for k in RSNA_ALL_LABELS:
                if k in adjusted:
                    adjusted[k] = np.clip(adjusted[k] * 0.95, 1e-6, 1-1e-6)
    
    return adjusted

def find_files_recursive(root_dir: str, filenames: List[str]) -> List[str]:
    """Recursively find files with given names in a directory."""
    found = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file in filenames:
                found.append(os.path.join(root, file))
    return found

def preflight_check(dataset_dir: str, project_dir: str, log_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Check dataset and project structure before running the submission pipeline.
    
    Args:
        dataset_dir: Path to the dataset directory
        project_dir: Path to the project directory
        log_path: Optional path to log file
        
    Returns:
        Dictionary with preflight check results
    """
    results = {
        'dataset_exists': os.path.exists(dataset_dir),
        'project_exists': os.path.exists(project_dir) if project_dir else False,
        'series_root': '',
        'test_csv': '',
        'series_count': 0,
        'scripts_found': [],
        'importable': False,
        'warnings': []
    }
    
    # Check dataset structure
    series_root = os.path.join(dataset_dir, 'series')
    results['series_root'] = series_root
    
    # Look for test CSV in common locations
    test_csvs = [
        os.path.join(dataset_dir, 'test.csv'),
        os.path.join(dataset_dir, 'sample_submission.csv'),
        os.path.join(dataset_dir, 'kaggle_evaluation', 'test.csv')
    ]
    
    for csv_path in test_csvs:
        if os.path.exists(csv_path):
            results['test_csv'] = csv_path
            break
    
    # Count series directories
    if os.path.exists(series_root):
        try:
            series_dirs = [d for d in os.listdir(series_root) 
                         if os.path.isdir(os.path.join(series_root, d))]
            results['series_count'] = len(series_dirs)
            results['series_sample'] = series_dirs[:3] if series_dirs else []
        except Exception as e:
            results['warnings'].append(f"Could not list series: {e}")
    
    # Check project structure
    if project_dir and os.path.exists(project_dir):
        # Look for common script names
        script_names = ['predict.py', 'run_inference_rc2.py', 'inference.py', 'main.py']
        results['scripts_found'] = find_files_recursive(project_dir, script_names)
        
        # Check if the project is importable
        try:
            import importlib.util
            spec = importlib.util.find_spec('rsna_aneurysm')
            results['importable'] = spec is not None
        except ImportError:
            pass
    
    # Log results
    log("\n=== PREFLIGHT: Dataset & Project Layout ===", log_path)
    log(f"Dataset exists: {results['dataset_exists']}", log_path)
    log(f"Series root exists: {os.path.exists(series_root)} at {series_root}", log_path)
    log(f"Test CSV found: {bool(results['test_csv'])} ({results['test_csv'] or 'none'})", log_path)
    
    if results.get('series_sample'):
        log(f"Series count: {results['series_count']}; sample: {results['series_sample']}", log_path)
    
    if project_dir:
        log(f"\nProject exists: {results['project_exists']} at {project_dir}", log_path)
        log(f"Importable? rsna_aneurysm in path: {results['importable']}", log_path)
        log(f"CLI scripts found: {len(results['scripts_found'])}", log_path)
        for i, script in enumerate(results['scripts_found'][:3], 1):
            log(f"  {i}. {script}", log_path)
        if len(results['scripts_found']) > 3:
            log(f"  ... and {len(results['scripts_found']) - 3} more", log_path)
    
    if results['warnings']:
        log("\nWarnings:", log_path)
        for warning in results['warnings']:
            log(f"  - {warning}", log_path)
    
    log("=" * 80 + "\n", log_path)
    return results

def validate_submission_format(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the submission DataFrame has the correct format.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, errors) where errors is a list of error messages
    """
    errors = []
    
    # Check for required columns
    required_columns = set(['SeriesInstanceUID'] + RSNA_ALL_LABELS)
    actual_columns = set(df.columns)
    
    # Check for missing columns
    missing = required_columns - actual_columns
    if missing:
        errors.append(f"Missing required columns: {missing}")
    
    # Check for invalid values
    for col in RSNA_ALL_LABELS + ['SeriesInstanceUID']:
        if col not in df.columns:
            continue
            
        if df[col].isna().any():
            errors.append(f"Column {col} contains NaN values")
            
        if col != 'SeriesInstanceUID':
            if (df[col] < 0).any() or (df[col] > 1).any():
                errors.append(f"Column {col} contains values outside [0, 1] range")
    
    # Check for duplicate SeriesInstanceUIDs
    if 'SeriesInstanceUID' in df.columns:
        dupes = df[df.duplicated('SeriesInstanceUID', keep=False)]
        if not dupes.empty:
            errors.append(f"Duplicate SeriesInstanceUIDs found: {dupes['SeriesInstanceUID'].tolist()[:5]}{'...' if len(dupes) > 5 else ''}")
    
    return len(errors) == 0, errors

def prepare_submission(predictions: Dict[str, List[float]], 
                      series_ids: List[str],
                      test_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Convert model predictions to a submission DataFrame with proper formatting.
    
    Args:
        predictions: Dictionary of predictions for each class
        series_ids: List of series instance IDs
        test_df: Optional test DataFrame to match series order
        
    Returns:
        DataFrame in submission format
    """
    # Create DataFrame from predictions
    pred_df = pd.DataFrame(predictions)
    
    # Add SeriesInstanceUID if not already present
    if 'SeriesInstanceUID' not in pred_df.columns:
        pred_df['SeriesInstanceUID'] = series_ids
    
    # Canonicalize column names
    pred_df = canonicalize_columns(pred_df)
    
    # If we have a test DataFrame, ensure we match its SeriesInstanceUID order
    if test_df is not None and 'SeriesInstanceUID' in test_df.columns:
        # Create a mapping from SeriesInstanceUID to index in test_df
        test_series = test_df['SeriesInstanceUID'].values
        test_order = {uid: i for i, uid in enumerate(test_series)}
        
        # Add a temporary column for sorting
        pred_df['__sort_order'] = pred_df['SeriesInstanceUID'].map(
            lambda x: test_order.get(x, len(test_series) + 1)
        )
        
        # Sort to match test_df order
        pred_df = pred_df.sort_values('__sort_order')
        pred_df = pred_df.drop('__sort_order', axis=1)
        
        # Ensure all test series are included
        missing = set(test_series) - set(pred_df['SeriesInstanceUID'])
        if missing:
            logger.warning(f"Adding {len(missing)} missing series with default predictions")
            missing_df = pd.DataFrame({
                'SeriesInstanceUID': list(missing)
            })
            for col in RSNA_ALL_LABELS:
                missing_df[col] = 1e-6
            pred_df = pd.concat([pred_df, missing_df], ignore_index=True)
            
            # Re-sort to ensure correct order
            pred_df['__sort_order'] = pred_df['SeriesInstanceUID'].map(
                lambda x: test_order.get(x, len(test_series) + 1)
            )
            pred_df = pred_df.sort_values('__sort_order')
            pred_df = pred_df.drop('__sort_order', axis=1)
    
    # Ensure all required columns exist with default values
    for label in RSNA_ALL_LABELS:
        if label not in pred_df.columns:
            pred_df[label] = 1e-6
    
    # Ensure logical consistency between presence and location predictions
    pred_df = reconcile_presence(pred_df)
    
    # Reorder columns to match expected format
    column_order = ['SeriesInstanceUID'] + RSNA_ALL_LABELS
    return pred_df[column_order]

def get_series_seed(series_id: str, max_seed: int = 2**32 - 1) -> int:
    """
    Generate a deterministic seed from a series ID.
    
    Args:
        series_id: SeriesInstanceUID
        max_seed: Maximum seed value
        
    Returns:
        Integer seed value
    """
    # Use MD5 hash of the series ID to get a deterministic but well-distributed seed
    hash_obj = hashlib.md5(series_id.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int % max_seed
