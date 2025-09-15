"""
Tests for the offline inference runner.
"""
import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.offline_runner import (
    run_offline_inference,
    _stable_series_seed,
    _is_dicom_file,
    validate_frame
)
from src.constants import RSNA_ALL_LABELS, REQUIRED_COLUMNS

# Test data
SAMPLE_SERIES_ID = "1.2.3.4"
SAMPLE_PREDICTIONS = {
    "SeriesInstanceUID": SAMPLE_SERIES_ID,
    **{label: 0.1 for label in RSNA_ALL_LABELS},
    "Aneurysm Present": 0.2
}


def test_stable_series_seed():
    """Test that the seed is stable for the same input."""
    seed1 = _stable_series_seed("test_series")
    seed2 = _stable_series_seed("test_series")
    assert seed1 == seed2
    
    # Different inputs should produce different seeds
    seed3 = _stable_series_seed("different_series")
    assert seed1 != seed3


def test_is_dicom_file(tmp_path):
    """Test DICOM file detection."""
    # Create a dummy DICOM file with magic number
    dicom_file = tmp_path / "test.dcm"
    with open(dicom_file, "wb") as f:
        f.write(b"\0" * 128 + b"DICM")
    
    assert _is_dicom_file(dicom_file)
    
    # Non-existent file
    assert not _is_dicom_file(tmp_path / "nonexistent")
    
    # Non-DICOM file
    not_dicom = tmp_path / "not_dicom.txt"
    not_dicom.write_text("not a DICOM file")
    assert not _is_dicom_file(not_dicom)


def test_validate_frame():
    """Test validation of prediction frames."""
    import pandas as pd
    
    # Valid frame
    valid_df = pd.DataFrame([SAMPLE_PREDICTIONS])
    is_valid, errors = validate_frame(valid_df)
    assert is_valid
    assert not errors
    
    # Missing required column
    invalid_df = valid_df.drop(columns=["Aneurysm Present"])
    is_valid, errors = validate_frame(invalid_df)
    assert not is_valid
    assert any("Missing required columns" in e for e in errors)
    
    # Out of range values
    invalid_df = valid_df.copy()
    invalid_df.iloc[0, 1] = 1.1  # First prediction column
    is_valid, errors = validate_frame(invalid_df)
    assert not is_valid
    assert any("outside [0, 1] range" in e for e in errors)


@patch("src.api.offline_runner._load_compact3d_predictor")
def test_run_offline_integration(mock_load_predictor, tmp_path):
    """Test the full offline inference pipeline with a mock predictor."""
    # Set up test data
    series_root = tmp_path / "series_root"
    series_dir = series_root / SAMPLE_SERIES_ID
    series_dir.mkdir(parents=True)
    
    # Create a dummy DICOM file
    dicom_file = series_dir / "1.dcm"
    with open(dicom_file, "wb") as f:
        f.write(b"\0" * 128 + b"DICM")
    
    # Set up mock predictor
    mock_predict = MagicMock(return_value={
        label: 0.1 for label in RSNA_ALL_LABELS
    })
    mock_load_predictor.return_value = (mock_predict, "test_predictor")
    
    # Run inference
    output_file = tmp_path / "submission.csv"
    result = run_offline_inference(
        series_root=series_root,
        output_path=output_file,
        limit=1
    )
    
    # Verify results
    assert output_file.exists()
    assert len(result) == 1
    assert result["SeriesInstanceUID"][0] == SAMPLE_SERIES_ID
    
    # Verify all required columns are present
    assert all(col in result.columns for col in REQUIRED_COLUMNS)
    
    # Verify predictor was called with the correct path
    mock_predict.assert_called_once()
    assert str(series_dir) in mock_predict.call_args[0][0]


def test_offline_runner_cli(tmp_path):
    """Test the command-line interface."""
    from src.api.offline_runner import main as offline_main
    
    # Create a test series directory
    series_root = tmp_path / "test_series"
    series_dir = series_root / SAMPLE_SERIES_ID
    series_dir.mkdir(parents=True)
    
    # Create a dummy DICOM file
    dicom_file = series_dir / "1.dcm"
    with open(dicom_file, "wb") as f:
        f.write(b"\0" * 128 + b"DICM")
    
    # Test with --help
    with patch("argparse.ArgumentParser._print_message") as mock_print:
        with patch("sys.argv", ["offline_runner.py", "--help"]):
            try:
                offline_main()
            except SystemExit:
                pass
            mock_print.assert_called()
    
    # Test with actual arguments
    output_file = tmp_path / "output.csv"
    with patch("sys.argv", [
        "offline_runner.py",
        "--series-root", str(series_root),
        "--output", str(output_file),
        "--limit", "1"
    ]):
        with patch("src.api.offline_runner.run_offline_inference") as mock_run:
            mock_run.return_value = "mocked_result"
            offline_main()
            mock_run.assert_called_once_with(
                series_root=str(series_root),
                output_path=str(output_file),
                limit=1,
                calibrate=True
            )


if __name__ == "__main__":
    pytest.main([__file__])
