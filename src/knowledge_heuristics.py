#!/usr/bin/env python3
"""
RSNA Intracranial Aneurysm Detection - Knowledge-Based Heuristics

Domain-specific heuristics derived from medical knowledge and training data:
- Anatomical location priors
- Imaging modality adjustments
- Clinical context integration
- Quality-based confidence weighting
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pydicom
from pydicom.dataset import FileDataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from src.utils.prediction_utils import RSNA_ALL_LABELS, RSNA_LOCATION_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeHeuristics:
    """Knowledge-based heuristics for aneurysm detection."""
    
    def __init__(self, knowledge_data_path: Optional[str] = None):
        """Initialize with domain knowledge."""
        
        # Anatomical location prevalence (from medical literature)
        self.location_prevalence = {
            'Anterior Communicating Artery': 0.325,  # 30-35%
            'Left Posterior Communicating Artery': 0.325,
            'Right Posterior Communicating Artery': 0.325,
            'Left Middle Cerebral Artery': 0.10,
            'Right Middle Cerebral Artery': 0.10,
            'Basilar Tip': 0.05,
            'Other Posterior Circulation': 0.02,
            'Left Infraclinoid Internal Carotid Artery': 0.05,
            'Right Infraclinoid Internal Carotid Artery': 0.05,
            'Left Supraclinoid Internal Carotid Artery': 0.08,
            'Right Supraclinoid Internal Carotid Artery': 0.08,
            'Left Anterior Cerebral Artery': 0.03,
            'Right Anterior Cerebral Artery': 0.03,
        }
        
        # Clinical risk factors and multipliers
        self.clinical_multipliers = {
            'high_risk_locations': {
                'Anterior Communicating Artery': 1.2,
                'Left Posterior Communicating Artery': 1.2,
                'Right Posterior Communicating Artery': 1.2,
                'Basilar Tip': 1.3,  # Higher risk of giant aneurysms
            },
            'modality_preferences': {
                'CTA': {'arterial_phase': 1.1, 'venous_phase': 0.92},
                'MRA': {'TOF': 1.0, 'contrast': 1.05},
                'DSA': {'standard': 1.2}  # Gold standard
            },
            'age_factors': {
                'young': (0, 30, 0.7),    # Lower risk in young patients
                'middle': (30, 60, 1.0),  # Standard risk
                'elderly': (60, 100, 1.2) # Higher risk in elderly
            },
            'gender_factors': {
                'F': 1.1,  # Slightly higher risk in females
                'M': 0.95
            }
        }
        
        # Quality assessment factors
        self.quality_factors = {
            'motion_artifacts': 0.85,
            'poor_contrast': 0.90,
            'incomplete_coverage': 0.80,
            'high_noise': 0.88,
            'optimal_quality': 1.0
        }
        
        # Load additional knowledge from processed data
        if knowledge_data_path:
            self.load_knowledge_data(knowledge_data_path)
    
    def load_knowledge_data(self, data_path: str):
        """Load processed knowledge from JSONL files."""
        data_path = Path(data_path)
        
        # Load knowledge dataset
        knowledge_file = data_path / 'processed' / 'knowledge_dataset.jsonl'
        if knowledge_file.exists():
            self.domain_knowledge = []
            with open(knowledge_file, 'r') as f:
                for line in f:
                    try:
                        self.domain_knowledge.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {len(self.domain_knowledge)} knowledge entries")
            self._extract_heuristics_from_knowledge()
    
    def _extract_heuristics_from_knowledge(self):
        """Extract actionable heuristics from domain knowledge."""
        for entry in self.domain_knowledge:
            if 'heuristics' in entry:
                for heuristic in entry['heuristics']:
                    # Apply heuristic rules to multipliers
                    if 'if' in heuristic and 'then' in heuristic:
                        condition = heuristic['if']
                        action = heuristic['then']
                        
                        # Example: {"if": "phase==arterial", "then": {"aneurysm_present_mult": 1.1}}
                        if 'aneurysm_present_mult' in action:
                            if 'arterial' in condition:
                                self.clinical_multipliers['modality_preferences']['CTA']['arterial_phase'] = action['aneurysm_present_mult']
                            elif 'venous' in condition:
                                self.clinical_multipliers['modality_preferences']['CTA']['venous_phase'] = action['aneurysm_present_mult']
    
    def apply_heuristics(self, 
                        predictions: pd.DataFrame,
                        metadata: Optional[Dict[str, Any]] = None,
                        dicom_path: Optional[str] = None) -> pd.DataFrame:
        """
        Apply knowledge-based heuristics to model predictions.
        
        Args:
            predictions: Raw model predictions
            metadata: DICOM metadata dictionary
            dicom_path: Path to DICOM files for metadata extraction
            
        Returns:
            Adjusted predictions with heuristics applied
        """
        adjusted_predictions = predictions.copy()
        
        # Extract metadata if not provided
        if metadata is None and dicom_path:
            metadata = self._extract_dicom_metadata(dicom_path)
        
        if metadata is None:
            logger.warning("No metadata available for heuristic adjustment")
            return adjusted_predictions
        
        # Apply location-based priors
        adjusted_predictions = self._apply_location_priors(adjusted_predictions)
        
        # Apply modality-specific adjustments
        adjusted_predictions = self._apply_modality_adjustments(adjusted_predictions, metadata)
        
        # Apply demographic adjustments
        adjusted_predictions = self._apply_demographic_adjustments(adjusted_predictions, metadata)
        
        # Apply quality-based adjustments
        adjusted_predictions = self._apply_quality_adjustments(adjusted_predictions, metadata)
        
        # Ensure consistency between presence and location predictions
        adjusted_predictions = self._enforce_consistency(adjusted_predictions)
        
        return adjusted_predictions
    
    def _extract_dicom_metadata(self, dicom_path: str) -> Dict[str, Any]:
        """Extract relevant metadata from DICOM files."""
        metadata = {}
        
        try:
            dicom_files = list(Path(dicom_path).glob("**/*.dcm"))
            if not dicom_files:
                return metadata
            
            # Read first DICOM for metadata
            ds = pydicom.dcmread(str(dicom_files[0]))
            
            # Basic demographics
            metadata['patient_age'] = getattr(ds, 'PatientAge', '').strip('Y') or '50'
            metadata['patient_sex'] = getattr(ds, 'PatientSex', 'U')
            
            # Imaging parameters
            metadata['modality'] = getattr(ds, 'Modality', 'CT')
            metadata['series_description'] = getattr(ds, 'SeriesDescription', '').lower()
            metadata['study_description'] = getattr(ds, 'StudyDescription', '').lower()
            
            # Contrast and timing
            metadata['contrast_agent'] = getattr(ds, 'ContrastBolusAgent', '')
            metadata['window_center'] = getattr(ds, 'WindowCenter', 40)
            metadata['window_width'] = getattr(ds, 'WindowWidth', 400)
            
            # Quality indicators
            metadata['slice_thickness'] = float(getattr(ds, 'SliceThickness', 1.0))
            metadata['pixel_spacing'] = getattr(ds, 'PixelSpacing', [1.0, 1.0])
            
            # Determine phase from series description
            series_desc = metadata['series_description']
            if 'arterial' in series_desc or 'art' in series_desc:
                metadata['phase'] = 'arterial'
            elif 'venous' in series_desc or 'ven' in series_desc:
                metadata['phase'] = 'venous'
            elif 'delay' in series_desc:
                metadata['phase'] = 'delayed'
            else:
                metadata['phase'] = 'unknown'
            
            # Quality assessment
            metadata['quality_score'] = self._assess_image_quality(ds, len(dicom_files))
            
        except Exception as e:
            logger.warning(f"Error extracting DICOM metadata: {e}")
        
        return metadata
    
    def _assess_image_quality(self, ds: FileDataset, num_slices: int) -> float:
        """Assess image quality based on DICOM parameters."""
        quality_score = 1.0
        
        # Slice thickness penalty
        slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))
        if slice_thickness > 3.0:
            quality_score *= 0.9
        elif slice_thickness > 5.0:
            quality_score *= 0.8
        
        # Number of slices
        if num_slices < 20:
            quality_score *= 0.85  # Too few slices
        elif num_slices > 200:
            quality_score *= 0.95  # Very high resolution
        
        # Pixel spacing
        pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        if isinstance(pixel_spacing, list) and len(pixel_spacing) >= 2:
            avg_spacing = (float(pixel_spacing[0]) + float(pixel_spacing[1])) / 2
            if avg_spacing > 1.0:
                quality_score *= 0.9
        
        return max(0.5, min(1.0, quality_score))
    
    def _apply_location_priors(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply anatomical location priors."""
        adjusted = predictions.copy()
        
        for location in RSNA_LOCATION_LABELS:
            if location in adjusted.columns:
                prior = self.location_prevalence.get(location, 0.05)
                
                # Bayesian update: P(aneurysm|prediction) ∝ P(prediction|aneurysm) * P(aneurysm)
                # Simple approximation: weighted average with prior
                weight = 0.1  # Weight of prior (10%)
                adjusted[location] = (1 - weight) * adjusted[location] + weight * prior
        
        return adjusted
    
    def _apply_modality_adjustments(self, predictions: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply modality-specific adjustments."""
        adjusted = predictions.copy()
        
        modality = metadata.get('modality', 'CT')
        phase = metadata.get('phase', 'unknown')
        
        # Get modality multiplier
        multiplier = 1.0
        if modality == 'CT':
            if phase == 'arterial':
                multiplier = self.clinical_multipliers['modality_preferences']['CTA']['arterial_phase']
            elif phase == 'venous':
                multiplier = self.clinical_multipliers['modality_preferences']['CTA']['venous_phase']
        elif modality == 'MR':
            if 'tof' in metadata.get('series_description', ''):
                multiplier = self.clinical_multipliers['modality_preferences']['MRA']['TOF']
            elif metadata.get('contrast_agent'):
                multiplier = self.clinical_multipliers['modality_preferences']['MRA']['contrast']
        
        # Apply to all location predictions
        for location in RSNA_LOCATION_LABELS:
            if location in adjusted.columns:
                adjusted[location] = adjusted[location] * multiplier
        
        return adjusted
    
    def _apply_demographic_adjustments(self, predictions: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply demographic-based adjustments."""
        adjusted = predictions.copy()
        
        # Age adjustment
        try:
            age = int(metadata.get('patient_age', 50))
            age_multiplier = 1.0
            
            for age_group, (min_age, max_age, multiplier) in self.clinical_multipliers['age_factors'].values():
                if min_age <= age < max_age:
                    age_multiplier = multiplier
                    break
        except (ValueError, TypeError):
            age_multiplier = 1.0
        
        # Gender adjustment
        gender = metadata.get('patient_sex', 'U')
        gender_multiplier = self.clinical_multipliers['gender_factors'].get(gender, 1.0)
        
        # Combined demographic multiplier
        demo_multiplier = age_multiplier * gender_multiplier
        
        # Apply to all predictions
        for location in RSNA_LOCATION_LABELS:
            if location in adjusted.columns:
                adjusted[location] = adjusted[location] * demo_multiplier
        
        return adjusted
    
    def _apply_quality_adjustments(self, predictions: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply image quality-based adjustments."""
        adjusted = predictions.copy()
        
        quality_score = metadata.get('quality_score', 1.0)
        
        # Lower quality reduces confidence in predictions
        quality_multiplier = 0.7 + 0.3 * quality_score  # Range: 0.7 to 1.0
        
        # Apply to all predictions
        for location in RSNA_LOCATION_LABELS:
            if location in adjusted.columns:
                adjusted[location] = adjusted[location] * quality_multiplier
        
        return adjusted
    
    def _enforce_consistency(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistency between presence and location predictions."""
        adjusted = predictions.copy()
        
        # Aneurysm Present should be at least the maximum of all location predictions
        location_cols = [col for col in RSNA_LOCATION_LABELS if col in adjusted.columns]
        
        if location_cols and 'Aneurysm Present' in adjusted.columns:
            for idx in adjusted.index:
                location_max = adjusted.loc[idx, location_cols].max()
                current_presence = adjusted.loc[idx, 'Aneurysm Present']
                
                # Ensure presence is at least as high as the maximum location
                adjusted.loc[idx, 'Aneurysm Present'] = max(current_presence, location_max)
        
        # Clip all predictions to [0, 1]
        for col in adjusted.columns:
            if col != 'SeriesInstanceUID':
                adjusted[col] = np.clip(adjusted[col], 0.0, 1.0)
        
        return adjusted
    
    def get_location_insights(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights about location-specific predictions."""
        insights = {}
        
        for location in RSNA_LOCATION_LABELS:
            if location in predictions.columns:
                preds = predictions[location].values
                
                insights[location] = {
                    'mean_prediction': float(np.mean(preds)),
                    'std_prediction': float(np.std(preds)),
                    'expected_prevalence': self.location_prevalence.get(location, 0.05),
                    'predicted_cases': int(np.sum(preds > 0.5)),
                    'high_confidence_cases': int(np.sum(preds > 0.8)),
                    'clinical_priority': self.clinical_multipliers['high_risk_locations'].get(location, 1.0)
                }
        
        return insights

def main():
    """Test the knowledge heuristics system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply Knowledge-Based Heuristics')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV file')
    parser.add_argument('--data-dir', type=str, 
                       default='/Users/augustolex/Desktop/RSNA Aneurysm Kaggle/data',
                       help='Path to data directory with knowledge files')
    parser.add_argument('--output', type=str, default='adjusted_predictions.csv',
                       help='Output path for adjusted predictions')
    
    args = parser.parse_args()
    
    # Load predictions
    predictions = pd.read_csv(args.predictions)
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Initialize heuristics
    heuristics = KnowledgeHeuristics(args.data_dir)
    
    # Apply heuristics (without DICOM metadata for this example)
    adjusted_predictions = heuristics.apply_heuristics(predictions)
    
    # Save adjusted predictions
    adjusted_predictions.to_csv(args.output, index=False)
    logger.info(f"Saved adjusted predictions to {args.output}")
    
    # Generate insights
    insights = heuristics.get_location_insights(adjusted_predictions)
    
    print("\n=== Location Insights ===")
    for location, data in insights.items():
        print(f"{location}:")
        print(f"  Mean Prediction: {data['mean_prediction']:.3f}")
        print(f"  Expected Prevalence: {data['expected_prevalence']:.3f}")
        print(f"  Predicted Cases: {data['predicted_cases']}")
        print(f"  High Confidence: {data['high_confidence_cases']}")

if __name__ == "__main__":
    main()
