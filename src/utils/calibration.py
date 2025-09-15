"""
Utility functions for applying calibration to model predictions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl
from scipy import interpolate

# RSNA labels (excluding 'Aneurysm Present' which is handled separately)
RSNA_LOCATIONS = [
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

class Calibrator:
    """Applies isotonic calibration to model predictions."""
    
    def __init__(self, calibration_dir: Union[str, Path]):
        """
        Initialize calibrator with calibration data from the specified directory.
        
        Args:
            calibration_dir: Directory containing calibration JSON files
        """
        self.calibration_dir = Path(calibration_dir)
        self.calibrators = {}
        
        # Load calibration data for each location
        for loc in RSNA_LOCATIONS:
            calib_file = self.calibration_dir / f"{loc.lower().replace(' ', '_')}_calibration.json"
            if calib_file.exists():
                with open(calib_file, 'r') as f:
                    calib_data = json.load(f)
                self.calibrators[loc] = interpolate.interp1d(
                    calib_data['x'],
                    calib_data['y'],
                    kind='linear',
                    bounds_error=False,
                    fill_value=(calib_data['y'][0], calib_data['y'][-1])
                )
    
    def calibrate_predictions(self, preds: Dict[str, float]) -> Dict[str, float]:
        """
        Apply calibration to a dictionary of predictions.
        
        Args:
            preds: Dictionary mapping location names to predicted probabilities
            
        Returns:
            Dictionary with calibrated probabilities
        """
        calibrated = {}
        
        # Calibrate each location
        for loc in RSNA_LOCATIONS:
            if loc in preds and loc in self.calibrators:
                # Clip to [0, 1] and apply calibration
                p = np.clip(preds[loc], 0, 1)
                calibrated[loc] = float(np.clip(self.calibrators[loc](p), 0, 1))
            elif loc in preds:
                # If no calibrator for this location, use original prediction
                calibrated[loc] = preds[loc]
        
        # Calculate 'Aneurysm Present' as 1 - product of (1 - p) for all locations
        if all(loc in calibrated for loc in RSNA_LOCATIONS):
            calibrated['Aneurysm Present'] = 1.0 - np.prod([1.0 - calibrated[loc] for loc in RSNA_LOCATIONS])
        elif 'Aneurysm Present' in preds:
            calibrated['Aneurysm Present'] = preds['Aneurysm Present']
        
        return calibrated
    
    def calibrate_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply calibration to a DataFrame of predictions.
        
        Args:
            df: DataFrame with prediction columns
            
        Returns:
            New DataFrame with calibrated predictions
        """
        if not self.calibrators:
            return df
            
        # Create a copy of the DataFrame to avoid modifying the original
        result = df.clone()
        
        # Calibrate each location
        for loc in RSNA_LOCATIONS:
            if loc in df.columns and loc in self.calibrators:
                # Clip to [0, 1] and apply calibration
                p = df[loc].clip(0, 1)
                calibrated = p.apply(lambda x: float(np.clip(self.calibrators[loc](x), 0, 1)))
                result = result.with_columns(pl.Series(loc, calibrated))
        
        # Recalculate 'Aneurysm Present' if all locations are present
        if all(loc in df.columns for loc in RSNA_LOCATIONS):
            # Calculate 1 - product of (1 - p) for all locations
            product = pl.lit(1.0)
            for loc in RSNA_LOCATIONS:
                product = product * (1.0 - result[loc])
            result = result.with_columns(
                pl.lit(1.0).sub(product).alias('Aneurysm Present')
            )
        
        return result

def load_calibrator(calibration_dir: Optional[Union[str, Path]] = None) -> Optional[Calibrator]:
    """
    Load calibrator from the specified directory.
    
    Args:
        calibration_dir: Directory containing calibration JSON files.
                        If None, returns None.
    """
    if calibration_dir is None:
        return None
    
    calibration_dir = Path(calibration_dir)
    if not calibration_dir.exists() or not any(calibration_dir.glob('*_calibration.json')):
        return None
        
    return Calibrator(calibration_dir)
