"""Tests for the submission validator."""

import os
import json
import tempfile
from pathlib import Path
import pytest
import polars as pl
import numpy as np

# Add the parent directory to the path to allow importing from src
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.validate_submission import validate_submission_file, validate_calibration_file

# Sample data for testing
SAMPLE_PREDICTIONS = {
    'SeriesInstanceUID': ['1.2.3', '4.5.6'],
    'Left Infraclinoid Internal Carotid Artery': [0.1, 0.2],
    'Right Infraclinoid Internal Carotid Artery': [0.3, 0.4],
    'Left Supraclinoid Internal Carotid Artery': [0.5, 0.6],
    'Right Supraclinoid Internal Carotid Artery': [0.7, 0.8],
    'Left Middle Cerebral Artery': [0.9, 0.1],
    'Right Middle Cerebral Artery': [0.2, 0.3],
    'Anterior Communicating Artery': [0.4, 0.5],
    'Left Anterior Cerebral Artery': [0.6, 0.7],
    'Right Anterior Cerebral Artery': [0.8, 0.9],
    'Left Posterior Communicating Artery': [0.1, 0.2],
    'Right Posterior Communicating Artery': [0.3, 0.4],
    'Basilar Tip': [0.5, 0.6],
    'Other Posterior Circulation': [0.7, 0.8],
    'Aneurysm Present': [0.9, 0.1]
}

# Sample calibration data
SAMPLE_CALIBRATION = {
    "x": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "y": [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
}

@pytest.fixture
def sample_submission_file():
    """Create a temporary submission file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pl.DataFrame(SAMPLE_PREDICTIONS)
        df.write_csv(f)
        f.flush()
        yield Path(f.name)
    if os.path.exists(f.name):
        os.unlink(f.name)

@pytest.fixture
def sample_calibration_file():
    """Create a temporary calibration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(SAMPLE_CALIBRATION, f)
        f.flush()
        yield Path(f.name)
    if os.path.exists(f.name):
        os.unlink(f.name)

def test_validate_good_submission(sample_submission_file):
    """Test validation of a correctly formatted submission file."""
    is_valid, errors = validate_submission_file(sample_submission_file)
    assert is_valid
    assert not errors

def test_validate_missing_column():
    """Test validation fails with missing columns."""
    # Create a DataFrame missing a required column
    data = SAMPLE_PREDICTIONS.copy()
    del data['Left Infraclinoid Internal Carotid Artery']
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv') as f:
        # Write the data to the file
        df = pl.DataFrame(data)
        df.write_csv(f)
        f.flush()
        
        is_valid, errors = validate_submission_file(Path(f.name))
        assert not is_valid
        # The error message might be about missing columns or empty CSV depending on how the file is written
        assert any("missing" in err.lower() or "empty" in err.lower() for err in errors)

def test_validate_out_of_range():
    """Test validation fails with out-of-range values."""
    # Create a DataFrame with an out-of-range value
    data = {k: v[:] for k, v in SAMPLE_PREDICTIONS.items()}
    data['Left Infraclinoid Internal Carotid Artery'] = [1.5, -0.1]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv') as f:
        # Write the data to the file
        df = pl.DataFrame(data)
        df.write_csv(f)
        f.flush()
        
        is_valid, errors = validate_submission_file(Path(f.name))
        assert not is_valid
        # Check for any error about out of range values
        assert any("outside" in err.lower() or "range" in err.lower() for err in errors)

def test_validate_calibration(sample_calibration_file):
    """Test validation of a valid calibration file."""
    is_valid, errors = validate_calibration_file(sample_calibration_file)
    assert is_valid
    assert not errors

def test_validate_invalid_calibration():
    """Test validation of an invalid calibration file."""
    # Create an invalid calibration (y values outside [0, 1])
    invalid_calib = {
        "x": [0.0, 1.0],
        "y": [0.0, 1.5]  # Invalid y value
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
        json.dump(invalid_calib, f)
        f.flush()
        
        is_valid, errors = validate_calibration_file(Path(f.name))
        assert not is_valid
        # Check for any error about values being out of range
        assert any("between 0 and 1" in err or "out of range" in err for err in errors)

def test_validate_calibration_mismatched_lengths():
    """Test validation fails when x and y arrays have different lengths."""
    invalid_calib = {
        "x": [0.0, 0.5, 1.0],
        "y": [0.0, 1.0]  # Different length
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
        json.dump(invalid_calib, f)
        f.flush()
        
        is_valid, errors = validate_calibration_file(Path(f.name))
        assert not is_valid
        assert "Length of 'x' and 'y' arrays must match" in errors[0]

def test_validate_calibration_non_increasing():
    """Test validation fails when x values are not non-decreasing."""
    invalid_calib = {
        "x": [0.0, 0.5, 0.4, 1.0],  # Not non-decreasing
        "y": [0.0, 0.5, 0.6, 1.0]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
        json.dump(invalid_calib, f)
        f.flush()
        
        is_valid, errors = validate_calibration_file(Path(f.name))
        assert not is_valid
        assert "'x' values must be non-decreasing" in errors[0]
