#!/usr/bin/env python3
"""
RSNA Aneurysm Detection - Submission Validator

Validates that a submission file meets the required format:
- Contains required columns (SeriesInstanceUID + 14 labels)
- All values are in [0, 1]
- No missing values
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import polars as pl
from jsonschema import validate, ValidationError

from .constants import (
    REQUIRED_COLUMNS, 
    RSNA_ALL_LABELS,
    ErrorMessages,
    DEFAULT_CALIBRATION_JSON,
    DEFAULT_SUBMISSION_CSV
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_submission_file(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate a submission file.
    
    Args:
        file_path: Path to the submission CSV file
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors: List[str] = []
    
    try:
        # Read the CSV file with proper error handling
        try:
            df = pl.read_csv(file_path)
        except Exception as e:
            error_msg = f"Error reading CSV file: {str(e)}"
            logger.error(error_msg)
            return False, [error_msg]
            
        if df.is_empty():
            error_msg = "CSV file is empty"
            logger.error(error_msg)
            return False, [error_msg]
            
        # Check required columns
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            error_msg = ErrorMessages.MISSING_COLUMNS.format(
                ", ".join(sorted(missing_cols))
            )
            errors.append(error_msg)
        
        extra_cols = set(df.columns) - set(REQUIRED_COLUMNS)
        if extra_cols:
            error_msg = ErrorMessages.UNEXPECTED_COLUMNS.format(
                ", ".join(sorted(extra_cols))
            )
            errors.append(error_msg)
        
        # If we're missing required columns, don't proceed with further validation
        if missing_cols:
            return False, errors
        
        # Check for missing values
        for col in RSNA_ALL_LABELS:  # Only check label columns
            null_count = df[col].null_count()
            if null_count > 0:
                error_msg = ErrorMessages.NULL_VALUES.format(col, null_count)
                errors.append(error_msg)
        
        # Check value ranges (only for probability columns)
        for col in RSNA_ALL_LABELS:  # Only check label columns
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            if min_val < 0 or max_val > 1:
                error_msg = ErrorMessages.OUT_OF_RANGE.format(
                    col, min_val, max_val
                )
                errors.append(error_msg)
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Submission file validation successful")
        else:
            logger.warning(f"Validation failed with {len(errors)} errors")
            
        return is_valid, errors
        
    except Exception as e:
        error_msg = f"Unexpected error during validation: {str(e)}"
        logger.exception(error_msg)
        return False, [error_msg]

def validate_calibration_file(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate a calibration JSON file against the schema.
    
    Args:
        file_path: Path to the calibration JSON file
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors: List[str] = []
    
    # Schema for the calibration file
    schema = {
        "type": "object",
        "properties": {
            "x": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 2
            },
            "y": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 2
            }
        },
        "required": ["x", "y"],
        "additionalProperties": False
    }
    
    try:
        # Load the calibration data
        with open(file_path) as f:
            calibration_data = json.load(f)
        
        # Validate against the schema
        validate(instance=calibration_data, schema=schema)
        
        # Additional validation: x and y must have the same length
        if len(calibration_data['x']) != len(calibration_data['y']):
            errors.append(ErrorMessages.CALIBRATION_LENGTH)
            return False, errors
            
        # Check that x values are non-decreasing
        x_values = calibration_data['x']
        if any(x_values[i] > x_values[i+1] for i in range(len(x_values)-1)):
            errors.append(ErrorMessages.CALIBRATION_ORDER)
            return False, errors
            
        logger.info("Calibration file validation successful")
        return True, []
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON: {str(e)}"
        logger.error(error_msg)
        return False, [error_msg]
    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        return False, [error_msg]
    except ValidationError as e:
        error_msg = f"Schema validation error: {str(e)}"
        logger.error(error_msg)
        return False, [error_msg]
    except Exception as e:
        error_msg = f"Unexpected error validating calibration: {str(e)}"
        logger.exception(error_msg)
        return False, [error_msg]

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate RSNA submission and calibration files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--submission', 
        type=str, 
        default=DEFAULT_SUBMISSION_CSV,
        help='Path to submission CSV file'
    )
    parser.add_argument(
        '--calibration', 
        type=str, 
        default=DEFAULT_CALIBRATION_JSON,
        help='Path to calibration JSON file'
    )
    parser.add_argument(
        '--verbose', 
        '-v', 
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()

def main() -> int:
    """
    Command-line interface for validation.
    
    Returns:
        int: 0 on success, non-zero on error
    """
    args = parse_arguments()
    
    # Configure logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    has_errors = False
    
    # Validate submission file if provided and exists
    submission_path = Path(args.submission)
    if submission_path.exists():
        logger.info(f"Validating submission file: {submission_path}")
        is_valid, errors = validate_submission_file(submission_path)
        if is_valid:
            print("✅ Submission file is valid")
        else:
            print("❌ Submission file validation failed:")
            for error in errors:
                print(f"  - {error}")
            has_errors = True
    elif args.submission != DEFAULT_SUBMISSION_CSV:  # Only warn if non-default path was provided
        logger.warning(f"Submission file not found: {submission_path}")
    
    # Validate calibration file if provided and exists
    calibration_path = Path(args.calibration)
    if calibration_path.exists():
        logger.info(f"Validating calibration file: {calibration_path}")
        is_valid, errors = validate_calibration_file(calibration_path)
        if is_valid:
            print("✅ Calibration file is valid")
        else:
            print("❌ Calibration file validation failed:")
            for error in errors:
                print(f"  - {error}")
            has_errors = True
    elif args.calibration != DEFAULT_CALIBRATION_JSON:  # Only warn if non-default path was provided
        logger.warning(f"Calibration file not found: {calibration_path}")
    
    # If no files were found, show usage
    if not submission_path.exists() and not calibration_path.exists():
        logger.error("No valid files found to validate")
        print("\nPlease provide at least one valid file to validate:")
        print(f"  --submission PATH   Path to submission CSV (default: {DEFAULT_SUBMISSION_CSV})")
        print(f"  --calibration PATH  Path to calibration JSON (default: {DEFAULT_CALIBRATION_JSON})")
        return 1
    
    return 1 if has_errors else 0

if __name__ == "__main__":
    sys.exit(main())
