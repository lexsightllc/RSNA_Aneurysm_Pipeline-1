from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import statistics
import pydicom

def _is_scout(series_description: str | None) -> bool:
    if not series_description:
        return False
    sd = series_description.lower()
    return "scout" in sd or "localizer" in sd

def _is_arterial(series_description: str | None) -> bool:
    if not series_description:
        return False
    sd = series_description.lower()
    return "arterial" in sd or "cta" in sd

def extract_dicom_metadata(series_dir: str | Path) -> Dict[str, Any]:
    """Extract metadata from a directory of DICOM files.
    
    Args:
        series_dir: Path to directory containing DICOM files
        
    Returns:
        Dictionary containing metadata about the DICOM series
    """
    series_dir = Path(series_dir)
    if not series_dir.exists():
        raise FileNotFoundError(f"{series_dir} does not exist")
        
    slice_thicknesses: List[float] = []
    arterial_flags: List[bool] = []
    count = 0
    
    for fp in sorted(series_dir.glob("*.dcm")):
        try:
            # Read DICOM file without pixel data for efficiency
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
        except Exception as e:
            print(f"Warning: Could not read {fp}: {e}")
            continue
            
        count += 1
        sd = getattr(ds, "SeriesDescription", None)
        st = getattr(ds, "SliceThickness", None)
        
        # Collect slice thickness (ignoring scout/localizer series)
        if st is not None and not _is_scout(sd):
            try:
                slice_thicknesses.append(float(st))
            except (ValueError, TypeError):
                pass
                
        # Check if this is an arterial phase series
        arterial_flags.append(_is_arterial(sd))
    
    # Calculate median slice thickness (ignoring None values)
    median_thickness = statistics.median(slice_thicknesses) if slice_thicknesses else None
    
    # Determine if this is likely an arterial phase series
    is_arterial = any(arterial_flags)
    
    return {
        "files_seen": count,
        "slice_thickness_mm": median_thickness,
        "is_arterial": is_arterial,
    }
