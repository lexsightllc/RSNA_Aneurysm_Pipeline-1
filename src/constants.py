"""
Centralized constants for the RSNA Aneurysm Detection project.

This module contains all shared constants, particularly the RSNA label definitions
that are used across multiple modules.
"""
from typing import List, Dict, Tuple, Set, Optional, Any, Union

# RSNA Location Labels
RSNA_LOCATION_LABELS: List[str] = [
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

# Special RSNA Labels
RSNA_ANEURYSM_PRESENT_LABEL: str = 'Aneurysm Present'

# Combined RSNA Labels (all locations + aneurysm present)
RSNA_ALL_LABELS: List[str] = RSNA_LOCATION_LABELS + [RSNA_ANEURYSM_PRESENT_LABEL]

# Submission file columns (SeriesInstanceUID + all labels)
RSNA_SUBMISSION_COLUMNS: List[str] = ['SeriesInstanceUID'] + RSNA_ALL_LABELS

# Required columns for submission validation
REQUIRED_COLUMNS: List[str] = RSNA_SUBMISSION_COLUMNS

# Default values for QA rules
class QaRules:
    """Default values for QA rules that can be overridden by knowledge base."""
    def __init__(self):
        # Default thresholds (can be overridden by knowledge base)
        self.cta_thin_max_mm: float = 1.5  # Maximum slice thickness for CTA to be considered thin
        self.min_series_size: int = 50  # Minimum number of slices for a valid series
        self.max_series_size: int = 1000  # Maximum number of slices for a valid series

# Default calibration parameters
DEFAULT_CALIBRATION: Dict[str, List[float]] = {
    "x": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "y": [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
}

# File paths and patterns
DEFAULT_KNOWLEDGE_CSV: str = "rsna_knowledge_dataset.csv"
DEFAULT_CALIBRATION_JSON: str = "calibration.json"
DEFAULT_SUBMISSION_CSV: str = "submission.csv"

# DICOM tags (commonly used in the project)
class DicomTags:
    SERIES_DESCRIPTION = (0x0008, 0x103E)
    STUDY_DESCRIPTION = (0x0008, 0x1030)
    IMAGE_TYPE = (0x0008, 0x0008)
    CONTRAST_BOLUS_AGENT = (0x0018, 0x0010)
    MODALITY = (0x0008, 0x0060)
    MANUFACTURER = (0x0008, 0x0070)
    INSTITUTION_NAME = (0x0008, 0x0080)
    SLICE_THICKNESS = (0x0018, 0x0050)
    ROWS = (0x0028, 0x0010)
    COLUMNS = (0x0028, 0x0011)

# Common strings for DICOM metadata processing
MODALITY_CT = 'CT'
MODALITY_MR = 'MR'
CONTRAST_AGENT = 'CONTRAST'

# Phase detection keywords (used in series description)
ARTERIAL_KEYWORDS = ['arterial', 'angio', 'cta', 'angio']
VENOUS_KEYWORDS = ['venous', 'delay', 'venosa', 'tardio']

# Validation error messages
class ErrorMessages:
    MISSING_COLUMNS = "Missing required columns: {}"
    UNEXPECTED_COLUMNS = "Unexpected columns: {}"
    NULL_VALUES = "Column '{}' contains {} null values"
    OUT_OF_RANGE = "Column '{}' contains values outside [0, 1] range (min={:.4f}, max={:.4f})"
    CALIBRATION_RANGE = "Calibration values must be between 0 and 1"
    CALIBRATION_LENGTH = "Length of 'x' and 'y' arrays must match"
    CALIBRATION_ORDER = "'x' values must be non-decreasing"
