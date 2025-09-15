#!/usr/bin/env python3
"""
RC2 inference runner for RSNA IAD (deterministic + heuristic-aware).

This script processes DICOM series, applies heuristics and calibration, and generates
submission files for the RSNA Intracranial Aneurysm Detection challenge.

Key Features:
- Predictor selection via command line or environment variables
- Scans series directories and validates DICOMs
- Writes submission.csv atomically
- Applies phase-aware heuristics (e.g., arterial↑, venous/delay↓) to 'Aneurysm Present'
- Supports calibration for probability outputs
"""
# Standard library imports
import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl
from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset

# Local application imports
from src.utils.calibration import load_calibrator
from src.constants import (
    RSNA_ALL_LABELS,
    REQUIRED_COLUMNS,
    DEFAULT_SUBMISSION_CSV,
    DEFAULT_CALIBRATION_JSON,
    RSNA_ANEURYSM_PRESENT_LABEL,
    RSNA_LOCATION_LABELS
)
from src.utils.predictor_loader import load_predictor_adapter, setup_environment

# Set up environment and logging
setup_environment()
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------- Knowledge / QA ----------------
@dataclass
class QaRules:
    cta_thin_max_mm: float = 1.2      # Recommended thin-slice CTA (table: 0.75-1.0 mm)
    thick_warn_mm: float = 1.5        # Above this, consider as inadequate geometry for CTA
    boost_loc_factor: float = 1.05    # Slight adjustment for locations
    boost_ap_factor: float = 1.08     # Slight adjustment for Aneurysm Present
    damp_factor: float = 0.95         # Slight downscale when quality is poor

@dataclass
class SeriesMeta:
    slice_thickness: Optional[float] = None
    modality: Optional[str] = None
    series_desc: Optional[str] = None

def load_knowledge_csv(path: Union[str, Path, None] = None) -> QaRules:
    """
    Loads knowledge CSV (if present) to confirm/adjust thresholds.
    Expected structure: columns [topic,key,value] (from the CSV you provided).
    If not found, returns safe defaults.
    """
    if not path:
        # Try default file in current working directory
        cand = Path("rsna_knowledge_dataset.csv")
        path = str(cand) if cand.exists() else None
    
    rules = QaRules()
    try:
        if path and Path(path).exists():
            import csv
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    k = (row.get("key") or "").lower()
                    v = (row.get("value") or "").strip()
                    # Simple mappings; keep safe if float conversion fails
                    if "cta_slice_recon_mm" in k:
                        try: 
                            rules.cta_thin_max_mm = float(v)
                        except (ValueError, TypeError):
                            pass
            print(f"[QA] Knowledge CSV loaded: {path}")
        else:
            print("[QA] Knowledge CSV not found; using defaults.")
    except Exception as e:
        print(f"[QA] Failed to read knowledge CSV ({path}): {e}")
    return rules

def read_dicom_meta(series_dir: Path) -> SeriesMeta:
    """Reads metadata from first found .dcm (fast, no pixel data)."""
    if dcmread is None:
        return SeriesMeta()
    try:
        for root, _, files in os.walk(series_dir):
            for fn in sorted(files):
                if fn.lower().endswith(".dcm"):
                    ds = dcmread(
                        os.path.join(root, fn), 
                        stop_before_pixels=True, 
                        force=True
                    )
                    st = None
                    try:
                        st = float(getattr(ds, "SliceThickness", None))
                    except (ValueError, TypeError):
                        st = None
                    mod = str(getattr(ds, "Modality", "") or "") or None
                    sdesc = str(getattr(ds, "SeriesDescription", "") or "") or None
                    return SeriesMeta(
                        slice_thickness=st,
                        modality=mod,
                        series_desc=sdesc
                    )
    except Exception as e:
        print(f"[QA] DICOM read failed for {series_dir.name}: {e}")
    return SeriesMeta()

def classify_geometry(meta: SeriesMeta, rules: QaRules) -> str:
    """
    Returns 'cta_thin', 'thick_or_unknown', or 'unknown'.
    """
    st = meta.slice_thickness
    if st is None:
        return "unknown"
    if st <= rules.cta_thin_max_mm:
        return "cta_thin"
    if st > rules.thick_warn_mm:
        return "thick_or_unknown"
    return "unknown"

def adjust_predictions(preds: Dict[str, float], geom_class: str, rules: QaRules) -> Dict[str, float]:
    """
    Applies LIGHT deterministic adjustments based on QA. Maintains clipping.
    """
    loc_labels = RSNA_ALL_LABELS[:-1]  # All except last (Aneurysm Present)
    ap_label = RSNA_ALL_LABELS[-1]     # Aneurysm Present
    
    clip = lambda x: float(min(max(x, 1e-3), 1 - 1e-3))
    out = preds.copy()
    
    # Allow disabling via environment variable
    if os.environ.get("RSNA_QA_ADJUST", "1").lower() not in {"1", "true"}:
        return out  # Adjustment disabled

    if geom_class == "cta_thin":
        # Boost predictions for thin-slice CTA
        for k in loc_labels:
            out[k] = clip(out[k] * rules.boost_loc_factor)
        out[ap_label] = clip(out[ap_label] * rules.boost_ap_factor)
    elif geom_class == "thick_or_unknown":
        # Slightly dampen predictions for thick/unknown slices
        for k in loc_labels + [ap_label]:
            out[k] = clip(out[k] * rules.damp_factor)
        # Maintain minimum consistency between AP and location predictions
        max_loc = max(out[k] for k in loc_labels)
        out[ap_label] = clip(max(out[ap_label], max_loc * 0.85))
    
    return out

# ---------- Constants ----------
# RSNA labels are imported from src.constants for consistency

# ---------- Helpers ----------
NATNUM = re.compile(r'(\d+)')
def natural_key(s: str) -> List:
    return [int(t) if t.isdigit() else t.lower() for t in NATNUM.split(str(s))]

def md5_of_file(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def has_dicoms(p: Path) -> bool:
    try:
        for _root, _dirs, files in os.walk(p):
            if any(f.lower().endswith('.dcm') for f in files):
                return True
    except Exception:
        pass
    return False

# ---------- Heuristics (opcional) ----------
def load_heuristics(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[heur] file not found: {path} (skipping)")
        return {}
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        print(f"[heur] loaded: {path}")
        return data
    except Exception as e:
        print(f"[heur] failed to load {path}: {e} (skipping)")
        return {}

def first_dicom_in(series_dir: Path) -> Optional[Path]:
    for root, _dirs, files in os.walk(series_dir):
        for f in sorted(files):
            if f.lower().endswith(".dcm"):
                return Path(root) / f
    return None

def extract_meta(series_dir: Path) -> Dict[str, Any]:
    """Extract DICOM metadata with enhanced error handling and type conversion.
    
    Returns:
        Dict containing DICOM metadata with standardized keys and converted values.
    """
    meta: Dict[str, Any] = {}
    dcm_path = first_dicom_in(series_dir)
    if not dcm_path:
        return meta
        
    try:
        import pydicom
        from pydicom.tag import Tag
        
        # Read only specific tags we care about for better performance
        tags_to_read = [
            Tag(0x0008, 0x103E),  # SeriesDescription
            Tag(0x0008, 0x1030),  # StudyDescription
            Tag(0x0008, 0x0008),  # ImageType
            Tag(0x0018, 0x0010),  # ContrastBolusAgent
            Tag(0x0008, 0x0060),  # Modality
            Tag(0x0008, 0x0070),  # Manufacturer
            Tag(0x0008, 0x0080),  # InstitutionName
            Tag(0x0018, 0x0050),  # SliceThickness
            Tag(0x0028, 0x0010),  # Rows
            Tag(0x0028, 0x0011),  # Columns
        ]
        
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, specific_tags=tags_to_read)
        
        # Standard metadata
        standard_fields = {
            'SeriesDescription': 'series_description',
            'StudyDescription': 'study_description',
            'ImageType': 'image_type',
            'ContrastBolusAgent': 'contrast',
            'Modality': 'modality',
            'Manufacturer': 'manufacturer',
            'InstitutionName': 'institution',
        }
        
        for dicom_key, meta_key in standard_fields.items():
            if hasattr(ds, dicom_key):
                val = getattr(ds, dicom_key, '')
                if isinstance(val, (list, tuple, pydicom.multival.MultiValue)):
                    val = ' '.join(str(v) for v in val if v)
                meta[meta_key] = str(val) if val is not None else ''
        
        # Numeric fields
        numeric_fields = {
            'SliceThickness': 'slice_thickness_mm',
            'Rows': 'rows',
            'Columns': 'cols'
        }
        
        for dicom_key, meta_key in numeric_fields.items():
            if hasattr(ds, dicom_key):
                try:
                    meta[meta_key] = float(getattr(ds, dicom_key, 0))
                except (ValueError, TypeError):
                    pass
        
        # Extract phase information from series description if available
        if 'series_description' in meta:
            sd = meta['series_description'].lower()
            meta['is_arterial'] = any(x in sd for x in ['arterial', 'angio', 'cta', 'angio'])
            meta['is_venous'] = any(x in sd for x in ['venous', 'delay', 'venosa', 'tardio'])
            
    except Exception as e:
        # Log error but don't fail
        meta['_error'] = f"Error reading DICOM: {str(e)[:100]}"
    
    return meta

def decide_actions(meta: Dict[str, Any], heuristics: Dict[str, Any]) -> Dict[str, float]:
    """Determine which heuristic actions to apply based on DICOM metadata.
    
    Args:
        meta: Dictionary of extracted DICOM metadata
        heuristics: Loaded heuristics configuration
        
    Returns:
        Dictionary of actions to apply, where keys are action names and values
        are multipliers or other parameters.
    """
    actions: Dict[str, float] = {}
    if not heuristics:
        return actions
        
    # Get phase rules from heuristics
    phase_rules = heuristics.get('phase_rules', [])
    
    # Apply phase-based rules
    for rule in phase_rules:
        # Check if this rule matches based on series description
        match_conditions = rule.get('match', {})
        matches = True
        
        # Check all match conditions
        for field, patterns in match_conditions.items():
            if field not in meta:
                matches = False
                break
                
            field_value = str(meta[field]).lower()
            if not any(p.lower() in field_value for p in patterns):
                matches = False
                break
                
        # If all conditions match, apply the actions
        if matches and 'actions' in rule:
            for action, value in rule['actions'].items():
                if not isinstance(value, (int, float)):
                    continue
                    
                # For multipliers, we combine them multiplicatively
                if action.endswith('_mult'):
                    actions[action] = actions.get(action, 1.0) * float(value)
                # For other actions, we take the first matching value
                elif action not in actions:
                    actions[action] = float(value)
    
    # Apply artifact suppression rules if present
    suppressors = heuristics.get('artifact_suppressors', [])
    for suppressor in suppressors:
        when = suppressor.get('when', {})
        should_suppress = True
        
        # Check all conditions for this suppressor
        for condition, required_value in when.items():
            # Special handling for boolean flags
            if isinstance(required_value, bool):
                if meta.get(condition) != required_value:
                    should_suppress = False
                    break
            # String matching for other conditions
            elif str(meta.get(condition, '')).lower() != str(required_value).lower():
                should_suppress = False
                break
                
        # If all conditions are met, apply the suppression
        if should_suppress:
            for field in suppressor.get('suppress', []):
                field_key = f"{field}_suppress"
                actions[field_key] = True
    
    return actions

def apply_actions(preds: Dict[str, float], actions: Dict[str, Any]) -> Dict[str, float]:
    """Apply heuristic actions to predictions.
    
    Args:
        preds: Dictionary of predictions to modify
        actions: Dictionary of actions to apply (from decide_actions)
        
    Returns:
        Modified predictions with heuristics applied
    """
    if not actions:
        return preds
        
    out = dict(preds)
    
    # Apply multipliers to predictions
    for action, value in actions.items():
        if not isinstance(value, (int, float)):
            continue
            
        # Handle aneurysm present multiplier
        if action == 'aneurysm_present_mult' and value != 1.0:
            out['Aneurysm Present'] = float(np.clip(
                out['Aneurysm Present'] * value, 
                1e-3, 1-1e-3
            ))
            
        # Handle volume multiplier (if present in predictions)
        elif action == 'min_volume_mm3_mult' and 'Minimum Volume (mm³)' in out:
            out['Minimum Volume (mm³)'] = float(np.clip(
                out['Minimum Volume (mm³)'] * value,
                0, 1e6  # Reasonable bounds for volume in mm³
            ))
    
    # Apply suppression rules (set probability to near-zero)
    for action, should_suppress in actions.items():
        if not isinstance(should_suppress, bool) or not should_suppress:
            continue
            
        # Handle field suppression (e.g., 'cortical_staining_suppress')
        if action.endswith('_suppress'):
            field = action[:-9]  # Remove '_suppress' suffix
            if field in out:
                out[field] = 1e-4  # Near-zero probability
    
    return out

# ---------- Predictor loader ----------
def load_predictor() -> Tuple[Any, str]:
    forced = os.environ.get('RC2_ADAPTER', '').strip().lower()
    if forced in {'enhanced', 'e'}:
        try:
            from inference_rc2_aneurysm_predictor import predict as _p
            return _p, 'enhanced (forced)'
        except ImportError as e:
            print(f"Error: forced Enhanced not found: {e}"); sys.exit(1)
    if forced in {'compact3d', 'c', 'compact'}:
        try:
            from adapter_compact3d_rc2 import predict as _p
            return _p, 'compact3d (forced)'
        except ImportError as e:
            print(f"Error: forced Compact-3D not found: {e}"); sys.exit(1)
    try:
        from inference_rc2_aneurysm_predictor import predict as _p
        return _p, 'enhanced (auto)'
    except ImportError:
        try:
            from adapter_compact3d_rc2 import predict as _p
            return _p, 'compact3d (auto)'
        except ImportError:
            print("Error: no predictor available (need inference_rc2_aneurysm_predictor.py or adapter_compact3d_rc2.py)")
            sys.exit(1)

def iter_series_dirs(series_root: str) -> List[Path]:
    """Iterate over valid DICOM series directories.
    
    Args:
        series_root: Root directory containing DICOM series subdirectories
        
    Returns:
        List of Path objects to valid DICOM series directories
        
    Exits if no valid DICOM series are found.
    """
    root = Path(series_root)
    if not root.exists() or not root.is_dir():
        logger.error(f"SERIES_ROOT not found or not a directory: {series_root}")
        sys.exit(1)
        
    dirs = [d for d in root.iterdir() if d.is_dir()]
    dirs.sort(key=lambda p: natural_key(p.name))
    valid = [d for d in dirs if has_dicoms(d)]
    
    if not valid:
        logger.error(f"No valid DICOM series found in {series_root}")
        sys.exit(1)
        
    return valid

# ---------- Main ----------
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RSNA IAD inference with optional heuristics and calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--series-root",
        type=str,
        required=True,
        help="Path to the directory containing DICOM series"
    )
    
    # Optional arguments
    parser.add_argument(
        "--heuristics-file",
        type=str,
        default="",
        help="Path to heuristics YAML file"
    )
    
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="",
        help="Path to calibration directory"
    )
    
    parser.add_argument(
        "--knowledge-csv",
        type=str,
        default="",
        help="Path to knowledge CSV file for QA rules"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_SUBMISSION_CSV,
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of series to process (0 for no limit)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main():
    """Main function to run RSNA IAD inference with optional heuristics and calibration."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting inference with series root: {args.series_root}")
    
    # Validate series root
    if not os.path.isdir(args.series_root):
        logger.error(f"Series root directory not found: {args.series_root}")
        sys.exit(1)
    
    # Load QA rules from knowledge base
    qa_rules = load_knowledge_csv(args.knowledge_csv or None)
    
    # Load heuristics if specified
    heuristics = None
    if args.heuristics_file and os.path.isfile(args.heuristics_file):
        heuristics = load_heuristics(args.heuristics_file)
        if heuristics:
            logger.info(f"Loaded heuristics from {args.heuristics_file}")
        else:
            logger.warning(f"No heuristics loaded from {args.heuristics_file}")
    
    # Load calibrator if calibration directory is specified
    calibrator = None
    if args.calibration_dir and os.path.isdir(args.calibration_dir):
        calibrator = load_calibrator(args.calibration_dir)
        if calibrator:
            logger.info(f"Using calibration from {args.calibration_dir}")
        else:
            logger.warning(f"No calibration files found in {args.calibration_dir}")
    
    # Initialize predictor
    predict_fn, predictor_name = load_predictor()
    logger.info(f"Using predictor: {predictor_name}")
    
    # Get series directories to process
    series_dirs = list(iter_series_dirs(args.series_root))
    total_series = len(series_dirs)
    
    # Apply limit if specified
    if args.limit > 0 and args.limit < total_series:
        series_dirs = series_dirs[:args.limit]
        logger.info(f"Limiting to first {args.limit} of {total_series} series")
    
    logger.info(f"Processing {len(series_dirs)} series with {predictor_name}...")
    
    # Track results and statistics
    results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    failed: List[str] = []
    
    # Main processing loop
    start_time = time.time()
    for i, sdir in enumerate(series_dirs, 1):
        sid = sdir.name
        logger.info(f"Processing [{i:3d}/{len(series_dirs)}] {sid}")
        iter_start = time.time()
        
        try:
            # Extract metadata for heuristics and QA
            meta = extract_meta(sdir)
            actions = decide_actions(meta, heuristics) if heuristics else {}
            
            # Get predictions
            df = predict_fn(str(sdir))
            preds = {col: float(df[col][0]) for col in RSNA_ALL_LABELS}
            
            # Apply calibration if available
            if calibrator:
                preds = calibrator.calibrate_predictions(preds)
            
            # Get DICOM metadata for QA
            meta = read_dicom_meta(sdir)
            geom = classify_geometry(meta, qa_rules)
            preds = adjust_predictions(preds, geom, qa_rules)
            
            # Apply heuristics if available
            if actions:
                preds = apply_actions(preds, actions)
            
            # Add to results
            row = {"SeriesInstanceUID": sid, **preds}
            results.append(row)
            
            # Log success
            iter_time = time.time() - iter_start
            latencies.append(iter_time)
            logger.debug(f"Completed {sid} in {iter_time:.2f}s")
            
        except Exception as e:
            iter_time = time.time() - iter_start
            logger.error(f"Failed to process {sid} after {iter_time:.2f}s: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            failed.append(sid)
    
    # Check if we have any results
    if not results:
        logger.error("No predictions generated successfully")
        sys.exit(1)
    
    # Create submission dataframe
    submission_cols = ["SeriesInstanceUID"] + RSNA_ALL_LABELS
    
    # Use polars if available (faster for large datasets)
    if pl is not None:
        submission = pl.DataFrame(results).select(submission_cols)
    else:
        submission = pd.DataFrame(results)[submission_cols]
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output files atomically
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp_path = tmp.name
    
    try:
        if pl is not None:
            submission.write_csv(tmp_path, float_precision=6)
        else:
            submission.to_csv(tmp_path, index=False, float_format="%.6f")
            
        os.replace(tmp_path, str(output_path))
        logger.info(f"Wrote {len(submission)} predictions to {output_path}")
    except Exception as e:
        logger.error(f"Error writing submission file: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        sys.exit(1)
    
    # Calculate statistics
    total_time = time.time() - start_time
    avg_latency = np.mean(latencies) if latencies else 0
    std_latency = np.std(latencies) if latencies else 0
    
    # Prepare summary
    summary = [
        "="*70,
        f"SUBMISSION SUMMARY ({predictor_name})",
        "="*70,
        f"Output: {output_path} ({len(results)}/{total_series} series)",
        f"Timing: {total_time:.1f}s total, {avg_latency:.2f} ± {std_latency:.2f}s per series",
        "",
        "Prediction ranges:"
    ]
    
    # Add prediction ranges
    if pl is not None:
        for col in RSNA_ALL_LABELS:
            col_data = submission[col].to_numpy()
            summary.append(f"  {col[:28]:28s}  min={col_data.min():.4f}  max={col_data.max():.4f}  mean={col_data.mean():.4f}")
        
        # Add first few predictions (convert to pandas for display)
        summary.extend([
            "",
            "First 2 predictions:",
            submission.head(2).to_pandas().to_string(),
            "-"*70,
            f"MD5: {md5_of_file(str(output_path))}"
        ])
    else:
        # Pandas DataFrame case
        for col in RSNA_ALL_LABELS:
            col_data = submission[col].values
            summary.append(f"  {col[:28]:28s}  min={col_data.min():.4f}  max={col_data.max():.4f}  mean={col_data.mean():.4f}")
        
        summary.extend([
            "",
            "First 2 predictions:",
            submission.head(2).to_string(),
            "-"*70,
            f"MD5: {md5_of_file(str(output_path))}"
        ])
    
    # Add failed series info if any
    if failed:
        summary.extend([
            "",
            f"WARNING: Failed to process {len(failed)} series:"
        ])
        for i, sid in enumerate(failed[:10], 1):
            summary.append(f"  {i:2d}. {sid}")
        if len(failed) > 10:
            summary.append(f"  ... and {len(failed) - 10} more")
    
    # Log the summary
    logger.info("\n".join(summary))
    
    # Show any heuristics that were applied
    if heuristics:
        logger.info("Heuristics applied:")
        for rule in heuristics.get('phase_rules', []):
            if 'match' in rule and 'actions' in rule:
                matches = ", ".join(f"{k}={v}" for k, v in rule.get('match', {}).items())
                actions = ", ".join(f"{k}={v}" for k, v in rule.get('actions', {}).items())
                print(f"  When {matches} → {actions}")
    print("="*70)

if __name__ == "__main__":
    main()
