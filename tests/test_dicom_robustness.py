import numpy as np
import pydicom
from pathlib import Path
from pydicom.dataset import FileDataset, FileMetaDataset
from src.metadata import extract_dicom_metadata

def _mk_dicom(path: Path, *, series_desc=None, slice_thickness=None, modality="CT"):
    """Helper function to create a minimal DICOM file for testing."""
    # Create minimal file meta info
    file_meta = FileMetaDataset()
    
    # Create a minimal DICOM dataset
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Set required DICOM attributes
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.Modality = modality
    
    # Set optional attributes if provided
    if series_desc is not None:
        ds.SeriesDescription = series_desc
    if slice_thickness is not None:
        ds.SliceThickness = str(slice_thickness)
    
    # Add minimal image attributes
    ds.Rows = 16  # Small size for test images
    ds.Columns = 16
    ds.BitsAllocated = 16
    ds.PixelRepresentation = 0  # Unsigned integer
    ds.SamplesPerPixel = 1  # Grayscale
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    # Add minimal pixel data (16x16 black image)
    ds.PixelData = np.zeros((16, 16), dtype=np.uint16).tobytes()
    
    # Save the DICOM file
    ds.save_as(path, write_like_original=False)

def test_median_thickness_and_phase_detection(tmp_path):
    """Test DICOM metadata extraction with various series types."""
    # Create a test directory with DICOM files
    sdir = tmp_path / "series"
    sdir.mkdir()
    
    # Create test DICOM files
    _mk_dicom(sdir / "a.dcm", series_desc="CTA Arterial", slice_thickness=1.0)
    _mk_dicom(sdir / "b.dcm", series_desc="CTA Arterial", slice_thickness=1.5)
    _mk_dicom(sdir / "c.dcm", series_desc="Scout", slice_thickness=10.0)  # Should be ignored for median
    
    # Extract metadata
    meta = extract_dicom_metadata(sdir)
    
    # Verify results
    assert meta["files_seen"] == 3, "Should process all DICOM files"
    assert meta["is_arterial"] is True, "Should detect arterial phase from series description"
    assert 1.0 <= meta["slice_thickness_mm"] <= 1.5, "Median thickness should be between 1.0 and 1.5 (scout excluded)"

def test_empty_directory(tmp_path):
    """Test behavior with an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    meta = extract_dicom_metadata(empty_dir)
    assert meta["files_seen"] == 0
    assert meta["slice_thickness_mm"] is None
    assert meta["is_arterial"] is False

def test_missing_metadata(tmp_path):
    """Test with DICOM files missing some metadata."""
    sdir = tmp_path / "minimal"
    sdir.mkdir()
    
    # Create minimal DICOM without series description or thickness
    _mk_dicom(sdir / "minimal.dcm")
    
    meta = extract_dicom_metadata(sdir)
    assert meta["files_seen"] == 1
    assert meta["slice_thickness_mm"] is None
    assert meta["is_arterial"] is False
