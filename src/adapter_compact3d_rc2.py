import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydicom import dcmread
from pydicom.dataset import FileDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model architecture
class Compact3DModel(nn.Module):
    def __init__(self, num_classes: int = 14):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(0.5)
        # Calculate correct input size after convolutions and pooling
        # Input: 128x128x128 -> conv1+pool -> 64x64x64 -> conv2+pool -> 32x32x32
        self.fc1 = nn.Linear(64 * 32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

# Global model instance
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 14

# Expected output columns
OUTPUT_COLUMNS = [
    'SeriesInstanceUID',
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

def load_model(weights_path: Optional[str] = None) -> nn.Module:
    """Load the model with optional weights."""
    model = Compact3DModel(num_classes=len(OUTPUT_COLUMNS)-1)  # -1 for SeriesInstanceUID
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from {weights_path}")
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            logger.warning("Initializing model with random weights")
    else:
        logger.warning("No weights provided, using random initialization")
    model.eval().to(DEVICE)
    return model

def apply_window(image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Apply windowing to DICOM image."""
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    
    windowed = np.clip(image, window_min, window_max)
    windowed = (windowed - window_min) / (window_max - window_min + 1e-6)
    return windowed

def load_dicom_series(directory: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load DICOM series from directory into a dictionary containing:
    - 'volume': 3D numpy array of pixel data
    - 'spacing': (x, y, z) voxel spacing in mm
    - 'window_center': Window center from DICOM metadata
    - 'window_width': Window width from DICOM metadata
    """
    directory = Path(directory)
    dicom_files = sorted(list(directory.glob("**/*.dcm")))
    
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {directory}")
    
    # Read first DICOM for metadata
    try:
        first_ds = dcmread(str(dicom_files[0]))
        window_center = float(getattr(first_ds, 'WindowCenter', 40))
        window_width = float(getattr(first_ds, 'WindowWidth', 400))
        
        # Get pixel spacing (default to 1.0 if not available)
        pixel_spacing = [
            float(getattr(first_ds, 'PixelSpacing', [1.0, 1.0])[0]),
            float(getattr(first_ds, 'PixelSpacing', [1.0, 1.0])[1]),
            float(getattr(first_ds, 'SliceThickness', 1.0))
        ]
    except Exception as e:
        logger.warning(f"Error reading DICOM metadata: {e}")
        window_center, window_width = 40, 400
        pixel_spacing = [1.0, 1.0, 1.0]
    
    # Load and sort slices by Z position
    slices = []
    z_positions = []
    
    for dcm_file in dicom_files:
        try:
            ds = dcmread(str(dcm_file))
            # Get Z position (ImagePositionPatient[2] if available, otherwise use filename order)
            z_pos = float(getattr(ds, 'ImagePositionPatient', [0, 0, len(slices)])[2])
            z_positions.append(z_pos)
            
            # Apply windowing if windowing parameters are available
            img = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept
            
            # Apply windowing
            img = apply_window(img, window_center, window_width)
            slices.append(img)
            
        except Exception as e:
            logger.warning(f"Error processing {dcm_file}: {e}")
    
    if not slices:
        raise ValueError("No valid DICOM slices found")
    
    # Sort slices by Z position
    if len(z_positions) == len(slices):
        slices = [s for _, s in sorted(zip(z_positions, slices), key=lambda x: x[0])]
    
    # Stack slices into 3D volume
    volume = np.stack(slices, axis=-1)
    
    return {
        'volume': volume,
        'spacing': pixel_spacing,
        'window_center': window_center,
        'window_width': window_width
    }

def resample_volume(volume: np.ndarray, original_spacing: List[float], target_spacing: List[float] = [1.0, 1.0, 1.0]) -> np.ndarray:
    """Resample volume to target voxel spacing using linear interpolation."""
    from scipy.ndimage import zoom
    
    if len(volume.shape) != 3:
        raise ValueError("Volume must be 3D")
    
    # Calculate zoom factors
    zoom_factors = [
        original_spacing[0] / target_spacing[0],
        original_spacing[1] / target_spacing[1],
        original_spacing[2] / target_spacing[2]
    ]
    
    # Apply zoom
    return zoom(volume, zoom_factors, order=1)  # order=1 for linear interpolation

def preprocess_volume(dicom_data: Dict[str, np.ndarray], target_size: tuple = (128, 128, 128)) -> torch.Tensor:
    """
    Preprocess volume for model input.
    
    Args:
        dicom_data: Dictionary containing 'volume', 'spacing', 'window_center', 'window_width'
        target_size: Target size for the output volume (depth, height, width)
        
    Returns:
        torch.Tensor: Preprocessed volume ready for model input
    """
    volume = dicom_data['volume']
    
    # Resample to isotropic voxels if needed
    if not all(np.isclose(dicom_data['spacing'], 1.0)):
        volume = resample_volume(volume, dicom_data['spacing'])
    
    # Pad or crop to target size
    def pad_or_crop(arr, target):
        if arr.shape[0] >= target:
            start = (arr.shape[0] - target) // 2
            return arr[start:start+target, :, :]
        else:
            pad_width = [(0, target - arr.shape[0])] + [(0, 0)] * (len(arr.shape) - 1)
            return np.pad(arr, pad_width, mode='constant')
    
    # Apply to each dimension
    for i, target in enumerate(target_size):
        volume = np.moveaxis(volume, i, 0)  # Move current dim to front
        volume = pad_or_crop(volume, target)
        volume = np.moveaxis(volume, 0, i)  # Move back
    
    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
    
    # Add channel and batch dimensions
    volume = np.expand_dims(volume, axis=0)  # Add channel dim
    volume = np.expand_dims(volume, axis=0)  # Add batch dim
    
    return torch.from_numpy(volume.astype(np.float32)).to(DEVICE)

def predict(series_dir: str, weights_path: Optional[str] = None, use_tta: bool = True) -> pd.DataFrame:
    """
    Main prediction function that loads DICOM series and returns predictions.
    
    Args:
        series_dir: Path to directory containing DICOM files
        weights_path: Path to model weights file
        use_tta: Whether to use test-time augmentation
        
    Returns:
        pd.DataFrame: DataFrame containing predictions for all output classes
    """
    global MODEL
    
    if MODEL is None:
        MODEL = load_model(weights_path)
    
    try:
        # Load DICOM data
        dicom_data = load_dicom_series(series_dir)
        
        def run_inference(volume_data, flip_axes=None):
            """Helper function to run inference with optional flipping."""
            if flip_axes:
                volume_data = {'volume': np.flip(volume_data['volume'], axis=flip_axes),
                              'spacing': volume_data['spacing'],
                              'window_center': volume_data['window_center'],
                              'window_width': volume_data['window_width']}
            
            input_tensor = preprocess_volume(volume_data)
            
            with torch.no_grad():
                outputs = MODEL(input_tensor)
                probs = outputs.cpu().numpy().flatten()
                
            # Flip predictions back if we flipped the input
            if flip_axes and 0 in flip_axes:  # Flip left/right predictions if we flipped along x
                probs = flip_lr_predictions(probs)
                
            return probs
        
        # Base prediction
        probs = run_inference(dicom_data)
        
        # Test-time augmentation (flip along different axes)
        if use_tta:
            tta_predictions = [probs]
            
            # Flip along x-axis (left/right)
            tta_predictions.append(run_inference(dicom_data, flip_axes=(0,)))
            
            # Flip along y-axis (anterior/posterior)
            tta_predictions.append(run_inference(dicom_data, flip_axes=(1,)))
            
            # Flip along z-axis (superior/inferior)
            tta_predictions.append(run_inference(dicom_data, flip_axes=(2,)))
            
            # Average predictions
            probs = np.mean(tta_predictions, axis=0)
        
        # Create results DataFrame
        result = {OUTPUT_COLUMNS[0]: [os.path.basename(series_dir)]}  # SeriesInstanceUID
        
        # Map predictions to output columns
        for i, col in enumerate(OUTPUT_COLUMNS[1:], 1):
            prob = probs[i-1] if i-1 < len(probs) else 0.0
            result[col] = [float(prob)]
        
        # Ensure 'Aneurysm Present' is the max of all location probabilities
        location_cols = [c for c in OUTPUT_COLUMNS[1:] if c != 'Aneurysm Present']
        if location_cols:
            location_probs = [result[col][0] for col in location_cols]
            result['Aneurysm Present'] = [float(max(location_probs))]
        
        return pd.DataFrame(result)
        
    except Exception as e:
        logger.error(f"Prediction failed for {series_dir}: {e}", exc_info=True)
        # Return zero probabilities if prediction fails
        return pd.DataFrame({col: [0.0 if i > 0 else str(series_dir)] 
                           for i, col in enumerate(OUTPUT_COLUMNS)})

def flip_lr_predictions(probs: np.ndarray) -> np.ndarray:
    """Flip left/right predictions to handle horizontal flips during TTA."""
    if len(probs) < 14:  # Ensure we have enough predictions
        return probs
    
    # Create a copy to avoid modifying the original
    flipped = probs.copy()
    
    # Define left/right pairs (0-based indices)
    lr_pairs = [
        (0, 1),   # Left/Right Infraclinoid ICA
        (2, 3),   # Left/Right Supraclinoid ICA
        (4, 5),   # Left/Right MCA
        (7, 8),   # Left/Right ACA
        (9, 10),  # Left/Right PCom
    ]
    
    # Swap left/right predictions
    for left_idx, right_idx in lr_pairs:
        if left_idx < len(flipped) and right_idx < len(flipped):
            flipped[left_idx], flipped[right_idx] = flipped[right_idx], flipped[left_idx]
    
    return flipped

def predict_wrapper(series_dir: str) -> pd.DataFrame:
    """Wrapper function that matches the expected signature for the inference script."""
    return predict(series_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-dir", type=str, required=True, help="Path to DICOM series directory")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    args = parser.parse_args()
    
    result = predict(args.series_dir, args.weights)
    print("Predictions:")
    print(result)