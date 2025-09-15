"""
Comprehensive tests for the RSNA Aneurysm Detection submission validator.

This module tests both submission CSV and calibration JSON validation,
including edge cases and error conditions.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

import click
import pytest
import pandas as pd
import numpy as np

# Add the parent directory to the path to allow importing from src
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.validate_submission import validate_submission_file, validate_calibration_file, main
from src.constants import (
    REQUIRED_COLUMNS,
    RSNA_ALL_LABELS,
    ErrorMessages,
    DEFAULT_CALIBRATION_JSON,
    DEFAULT_SUBMISSION_CSV
)

# Sample data for testing
SAMPLE_PREDICTIONS = {
    'SeriesInstanceUID': ['1.2.3', '4.5.6', '7.8.9'],
    'Left Infraclinoid Internal Carotid Artery': [0.1, 0.2, 0.15],
    'Right Infraclinoid Internal Carotid Artery': [0.3, 0.4, 0.35],
    'Left Supraclinoid Internal Carotid Artery': [0.5, 0.6, 0.55],
    'Right Supraclinoid Internal Carotid Artery': [0.7, 0.8, 0.75],
    'Left Middle Cerebral Artery': [0.9, 0.1, 0.5],
    'Right Middle Cerebral Artery': [0.2, 0.3, 0.25],
    'Anterior Communicating Artery': [0.4, 0.5, 0.45],
    'Left Anterior Cerebral Artery': [0.6, 0.7, 0.65],
    'Right Anterior Cerebral Artery': [0.8, 0.9, 0.85],
    'Left Posterior Communicating Artery': [0.1, 0.2, 0.15],
    'Right Posterior Communicating Artery': [0.3, 0.4, 0.35],
    'Basilar Tip': [0.5, 0.6, 0.55],
    'Other Posterior Circulation': [0.7, 0.8, 0.75],
    'Aneurysm Present': [0.9, 0.1, 0.5]
}

# Sample calibration data
SAMPLE_CALIBRATION = {
    "x": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "y": [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
}

# Invalid calibration data for testing
INVALID_CALIBRATION = {
    "x": [0.0, 0.1, 0.3, 0.2, 0.4],  # Not monotonically increasing
    "y": [0.0, 0.1, 0.2, 0.3, 0.4]
}

# Calibration with mismatched array lengths
MISMATCHED_CALIBRATION = {
    "x": [0.0, 0.1, 0.2, 0.3, 0.4],
    "y": [0.0, 0.1, 0.2]  # Different length
}

# Calibration with out-of-range values
OUT_OF_RANGE_CALIBRATION = {
    "x": [0.0, 0.1, 0.2, 0.3, 1.1],  # 1.1 is out of range
    "y": [0.0, 0.1, 0.2, 0.3, 0.4]
}

@pytest.fixture
def sample_submission_file() -> Path:
    """Create a temporary submission file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame(SAMPLE_PREDICTIONS)
        df.to_csv(f, index=False)
        f.flush()
        yield Path(f.name)
    # Cleanup is handled by the fixture's finalizer
    if Path(f.name).exists():
        Path(f.name).unlink()

@pytest.fixture
def empty_submission_file() -> Path:
    """Create an empty CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("")
        f.flush()
        yield Path(f.name)
    if Path(f.name).exists():
        Path(f.name).unlink()

@pytest.fixture
def invalid_csv_file() -> Path:
    """Create an invalid CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("not,a,valid,csv,file")
        f.flush()
        yield Path(f.name)
    if Path(f.name).exists():
        Path(f.name).unlink()

@pytest.fixture
def sample_calibration_file() -> Path:
    """Create a temporary calibration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(SAMPLE_CALIBRATION, f)
        f.flush()
        yield Path(f.name)
    if Path(f.name).exists():
        Path(f.name).unlink()

@pytest.fixture
def invalid_calibration_file() -> Path:
    """Create an invalid calibration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(INVALID_CALIBRATION, f)
        f.flush()
        yield Path(f.name)
    if Path(f.name).exists():
        Path(f.name).unlink()

@pytest.fixture
def mismatched_calibration_file() -> Path:
    """Create a calibration file with mismatched array lengths for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(MISMATCHED_CALIBRATION, f)
        f.flush()
        yield Path(f.name)
    if Path(f.name).exists():
        Path(f.name).unlink()

@pytest.fixture
def out_of_range_calibration_file() -> Path:
    """Create a calibration file with out-of-range values for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(OUT_OF_RANGE_CALIBRATION, f)
        f.flush()
        yield Path(f.name)
    if Path(f.name).exists():
        Path(f.name).unlink()

# Helper functions
def create_test_submission(
    missing_columns: List[str] = None,
    extra_columns: Dict[str, List[Any]] = None,
    out_of_range_columns: Dict[str, List[float]] = None,
    null_columns: Dict[str, List[bool]] = None
) -> Path:
    """
    Create a test submission file with specific issues for testing.
    
    Args:
        missing_columns: List of columns to remove
        extra_columns: Dict of {column_name: values} to add
        out_of_range_columns: Dict of {column_name: values} to replace with out-of-range values
        null_columns: Dict of {column_name: [bool]} indicating which values to set to None
        
    Returns:
        Path to the created test file
    """
    # Create a copy of the sample data
    data = SAMPLE_PREDICTIONS.copy()
    
    # Remove specified columns
    if missing_columns:
        for col in missing_columns:
            if col in data:
                del data[col]
    
    # Add extra columns
    if extra_columns:
        data.update(extra_columns)
    
    # Replace values with out-of-range values
    if out_of_range_columns:
        for col, values in out_of_range_columns.items():
            if col in data:
                data[col] = values
    
    # Set null values
    if null_columns:
        for col, mask in null_columns.items():
            if col in data:
                # Convert to pandas Series to handle None values properly
                s = pd.Series(data[col])
                s[mask] = None
                data[col] = s.tolist()
    
    # Create a temporary file with the modified data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame(data)
        df.to_csv(f, index=False)
        f.flush()
        return Path(f.name)

# Test cases for submission validation

def test_validate_good_submission(sample_submission_file):
    """Test validation of a correctly formatted submission file."""
    is_valid, errors = validate_submission_file(sample_submission_file)
    assert is_valid is True, f"Expected valid submission, but got errors: {errors}"
    assert not errors, f"Expected no errors, but got: {errors}"

def test_validate_empty_file(empty_submission_file):
    """Test validation of an empty submission file."""
    is_valid, errors = validate_submission_file(empty_submission_file)
    assert is_valid is False, "Expected invalid submission for empty file"
    # The error message might vary based on the CSV parser, so we'll check for any error
    assert errors, f"Expected an error message, got: {errors}"

def test_validate_invalid_csv(invalid_csv_file):
    """Test validation of an invalid CSV file."""
    is_valid, errors = validate_submission_file(invalid_csv_file)
    assert is_valid is False, "Expected invalid submission for invalid CSV"
    # The error message might vary, so we'll just check that there's an error
    assert errors, f"Expected an error message, got: {errors}"

def test_validate_missing_columns():
    """Test validation fails with missing columns."""
    # Create a submission with a missing column
    missing_col = 'Aneurysm Present'
    with create_test_submission(missing_columns=[missing_col]) as test_file:
        is_valid, errors = validate_submission_file(test_file)
        assert is_valid is False, "Expected invalid submission for missing column"
        assert f"Missing required columns: {missing_col}" in errors[0]

def test_validate_extra_columns():
    """Test validation fails with extra columns."""
    # Create a submission with an extra column
    extra_col = 'Extra_Column'
    # Make sure we have the same number of values as other columns
    with create_test_submission(extra_columns={extra_col: [1, 2, 3]}) as test_file:
        is_valid, errors = validate_submission_file(test_file)
        assert is_valid is False, "Expected invalid submission for extra column"
        assert any("Unexpected columns:" in err for err in errors), \
            f"Expected error about unexpected columns, got: {errors}"

def test_validate_out_of_range():
    """Test validation fails with out-of-range values."""
    # Create a submission with out-of-range values in a column
    col_name = 'Aneurysm Present'
    with create_test_submission(
        out_of_range_columns={col_name: [0.5, 1.5, -0.1]}
    ) as test_file:
        is_valid, errors = validate_submission_file(test_file)
        assert is_valid is False, "Expected invalid submission for out-of-range values"
        assert any("outside [0, 1] range" in err for err in errors), \
            f"Expected error about out of range values, got: {errors}"

def test_validate_missing_values():
    """Test validation fails with missing values."""
    # Create a submission with missing values in a column
    col_name = 'Aneurysm Present'
    # Make sure we have the same number of values as other columns
    with create_test_submission(
        null_columns={col_name: [False, True, False]}
    ) as test_file:
        is_valid, errors = validate_submission_file(test_file)
        assert is_valid is False, "Expected invalid submission for missing values"
        assert any("null values" in err for err in errors), \
            f"Expected error about null values, got: {errors}"

# Test cases for calibration validation

def test_validate_good_calibration(sample_calibration_file):
    """Test validation of a valid calibration file."""
    is_valid, errors = validate_calibration_file(sample_calibration_file)
    assert is_valid is True, f"Expected valid calibration, but got errors: {errors}"
    assert not errors, f"Expected no errors, but got: {errors}"

def test_validate_invalid_calibration(invalid_calibration_file):
    """Test validation fails with non-increasing x values."""
    is_valid, errors = validate_calibration_file(invalid_calibration_file)
    assert is_valid is False, "Expected invalid calibration for non-increasing x values"
    assert any("non-decreasing" in err.lower() or "monotonic" in err.lower() for err in errors), \
        f"Expected error about non-decreasing values, got: {errors}"

def test_validate_mismatched_calibration(mismatched_calibration_file):
    """Test validation fails with mismatched array lengths."""
    is_valid, errors = validate_calibration_file(mismatched_calibration_file)
    assert is_valid is False, "Expected invalid calibration for mismatched array lengths"
    assert "Length of 'x' and 'y' arrays must match" in errors[0], f"Unexpected errors: {errors}"

def test_validate_out_of_range_calibration(out_of_range_calibration_file):
    """Test validation fails with out-of-range values."""
    is_valid, errors = validate_calibration_file(out_of_range_calibration_file)
    assert is_valid is False, "Expected invalid calibration for out-of-range values"
    assert any("greater than the maximum" in err for err in errors), \
        f"Expected error about out-of-range values, got: {errors}"

def test_validate_nonexistent_file():
    """Test validation of a non-existent file."""
    non_existent = Path("/path/that/does/not/exist.json")
    is_valid, errors = validate_calibration_file(non_existent)
    assert is_valid is False, "Expected invalid calibration for non-existent file"
    assert "File not found" in errors[0], f"Unexpected errors: {errors}"

# Test command-line interface

# Mock the main function for CLI testing
@click.command()
@click.option('--submission', type=click.Path(exists=True), help='Path to submission CSV file')
@click.option('--calibration', type=click.Path(exists=True), help='Path to calibration JSON file')
def cli(submission, calibration):
    """CLI entry point for testing."""
    has_errors = False
    
    if submission:
        is_valid, errors = validate_submission_file(Path(submission))
        if is_valid:
            click.echo("✅ Submission file is valid")
        else:
            click.echo("❌ Submission file validation failed:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            has_errors = True
    
    if calibration:
        is_valid, errors = validate_calibration_file(Path(calibration))
        if is_valid:
            click.echo("✅ Calibration file is valid")
        else:
            click.echo("❌ Calibration file validation failed:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            has_errors = True
    
    if not submission and not calibration:
        click.echo("Error: Please provide at least one of --submission or --calibration")
        has_errors = True
    
    sys.exit(1 if has_errors else 0)

def test_cli_valid_submission(runner, sample_submission_file):
    """Test CLI with a valid submission file."""
    result = runner.invoke(
        cli, 
        ['--submission', str(sample_submission_file)]
    )
    assert result.exit_code == 0
    assert "Submission file is valid" in result.output

def test_cli_invalid_submission(runner, empty_submission_file):
    """Test CLI with an invalid submission file."""
    result = runner.invoke(
        cli,
        ['--submission', str(empty_submission_file)]
    )
    assert result.exit_code == 1
    assert "validation failed" in result.output.lower()

def test_cli_valid_calibration(runner, sample_calibration_file):
    """Test CLI with a valid calibration file."""
    result = runner.invoke(
        cli,
        ['--calibration', str(sample_calibration_file)]
    )
    assert result.exit_code == 0
    assert "Calibration file is valid" in result.output

def test_cli_invalid_calibration(runner, invalid_calibration_file):
    """Test CLI with an invalid calibration file."""
    result = runner.invoke(
        cli,
        ['--calibration', str(invalid_calibration_file)]
    )
    assert result.exit_code == 1
    assert "validation failed" in result.output.lower()

def test_cli_no_arguments(runner):
    """Test CLI with no arguments shows error."""
    result = runner.invoke(cli, [])
    assert result.exit_code == 1
    assert "Error: Please provide at least one of --submission or --calibration" in result.output

# Fixture for CLI testing
@pytest.fixture
def runner():
    """Fixture for CLI testing."""
    from click.testing import CliRunner
    return CliRunner()

def test_validate_calibration_non_increasing():
    """Test validation fails when x values are not non-decreasing."""
    # Create a calibration with non-increasing x values
    calib = {
        "x": [0.0, 0.2, 0.1, 0.3, 0.4],  # Not non-decreasing
        "y": [0.0, 0.2, 0.1, 0.3, 0.4]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
        json.dump(calib, f)
        f.flush()

        is_valid, errors = validate_calibration_file(Path(f.name))
        assert not is_valid
        assert any("non-decreasing" in err.lower() or "monotonic" in err.lower() for err in errors), \
            f"Expected error about non-decreasing values, got: {errors}"
