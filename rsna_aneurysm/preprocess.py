"""DICOM loading and preprocessing utilities."""
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def load_dicom_volume(series_path: Union[str, Path]) -> np.ndarray:
    """Load a DICOM series into a 3D numpy array.
    
    Args:
        series_path: Path to directory containing DICOM files for one series.
        
    Returns:
        3D numpy array with shape (depth, height, width).
    """
    series_path = Path(series_path)
    if not series_path.is_dir():
        raise ValueError(f"Series path {series_path} is not a directory")
    
    # Find and sort DICOM files
    dicom_files = []
    for f in series_path.glob('*'):
        try:
            if f.is_file() and f.suffix.lower() in ('.dcm', '.dicom', ''):
                dicom_files.append(f)
        except (IOError, PermissionError):
            continue
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_path}")
    
    # Sort files by instance number or filename
    def get_instance_number(f: Path) -> int:
        try:
            return int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
        except (AttributeError, ValueError, TypeError):
            return int(f.stem.split('_')[-1]) if f.stem.split('_')[-1].isdigit() else 0
    
    dicom_files.sort(key=get_instance_number)
    
    # Read and stack slices
    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f)
            if hasattr(ds, 'pixel_array'):
                pixel_array = ds.pixel_array
                
                # Apply VOI LUT if available
                if 'VOILUTSequence' in ds:
                    pixel_array = apply_voi_lut(pixel_array, ds)
                
                # Rescale slope and intercept if available
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    pixel_array = pixel_array.astype(np.float32) * ds.RescaleSlope + ds.RescaleIntercept
                
                slices.append(pixel_array)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
            continue
    
    if not slices:
        raise ValueError(f"No valid DICOM slices found in {series_path}")
    
    # Stack slices into 3D volume
    volume = np.stack(slices, axis=0)
    
    # Ensure consistent orientation (axial slices)
    if volume.shape[0] < volume.shape[1]:  # If depth is smaller than height/width, assume it's not the first dimension
        volume = np.transpose(volume, (1, 2, 0))
    
    return volume.astype(np.float32)


def preprocess_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int] = (96, 128, 128),
    window_center: float = 40.0,
    window_width: float = 400.0
) -> np.ndarray:
    """Preprocess a 3D volume for model input.
    
    Args:
        volume: Input 3D volume with shape (D, H, W).
        target_shape: Target shape (depth, height, width) for resizing.
        window_center: Center of the window for windowing.
        window_width: Width of the window for windowing.
        
    Returns:
        Preprocessed volume with shape (1, D', H', W') ready for model input.
    """
    import cv2
    
    # Apply windowing (CT window)
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    # Clip and normalize to [0, 1]
    volume = np.clip(volume, window_min, window_max)
    volume = (volume - window_min) / (window_max - window_min)
    
    # Resize to target shape
    depth, height, width = volume.shape
    target_depth, target_height, target_width = target_shape
    
    # Resize each slice
    resized_slices = []
    for i in range(depth):
        # Resize to target height and width
        resized = cv2.resize(
            volume[i], 
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )
        resized_slices.append(resized)
    
    # Stack resized slices
    volume = np.stack(resized_slices, axis=0)
    
    # Resize in depth dimension if needed
    if target_depth != depth:
        # Resize along depth using linear interpolation
        depth_scale = target_depth / depth
        volume = cv2.resize(
            volume.transpose(1, 2, 0),  # (H, W, D)
            (target_depth, target_height),  # (D, H)
            interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)  # Back to (D, H, W)
    
    # Add channel dimension and normalize
    volume = volume[np.newaxis, ...]  # (1, D, H, W)
    
    # Standardize
    mean = np.mean(volume)
    std = np.std(volume)
    if std > 0:
        volume = (volume - mean) / std
    
    return volume.astype(np.float32)


def load_and_preprocess_series(
    series_path: Union[str, Path],
    target_shape: Tuple[int, int, int] = (96, 128, 128),
    window_center: float = 40.0,
    window_width: float = 400.0
) -> np.ndarray:
    """Load a DICOM series and preprocess it for model input.
    
    Args:
        series_path: Path to directory containing DICOM files for one series.
        target_shape: Target shape (depth, height, width) for resizing.
        window_center: Center of the window for windowing.
        window_width: Width of the window for windowing.
        
    Returns:
        Preprocessed volume with shape (1, D', H', W') ready for model input.
    """
    # Load DICOM volume
    volume = load_dicom_volume(series_path)
    
    # Preprocess volume
    preprocessed = preprocess_volume(
        volume,
        target_shape=target_shape,
        window_center=window_center,
        window_width=window_width
    )
    
    return preprocessed
