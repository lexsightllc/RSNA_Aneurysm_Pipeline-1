import os
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import numpy as np
import pydicom

class RSNADataProcessor:
    def __init__(self, data_dir: str = "/kaggle/input/rsna-intracranial-aneurysm-detection"):
        """Initialize the data processor with the competition data directory."""
        self.data_dir = Path(data_dir)
        self.train_csv = self.data_dir / "train.csv"
        self.train_localizers = self.data_dir / "train_localizers.csv"
        self.series_dir = self.data_dir / "series"
        self.segmentations_dir = self.data_dir / "segmentations"
        
        # Validate paths
        self._validate_paths()
        
        # Load data
        self.train_df = pd.read_csv(self.train_csv)
        self.localizers_df = pd.read_csv(self.train_localizers)
        
        print(f"Loaded {len(self.train_df)} training samples")
        print(f"Loaded {len(self.localizers_df)} localizer samples")
    
    def _validate_paths(self) -> None:
        """Validate that all required data paths exist."""
        required_paths = [
            self.data_dir,
            self.train_csv,
            self.train_localizers,
            self.series_dir,
            self.segmentations_dir
        ]
        
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {path}")
    
    def explore_data(self) -> Dict:
        """Explore and return basic statistics about the training data."""
        stats = {
            "total_samples": len(self.train_df),
            "columns": list(self.train_df.columns),
            "missing_values": self.train_df.isnull().sum().to_dict()
        }
        
        # Check for aneurysm-related columns
        aneurysm_cols = [col for col in self.train_df.columns if 'aneurysm' in col.lower()]
        if aneurysm_cols:
            stats["aneurysm_columns"] = aneurysm_cols
            # Use the first aneurysm-related column found
            stats["aneurysm_distribution"] = self.train_df[aneurysm_cols[0]].value_counts().to_dict()
        else:
            stats["warning"] = "No aneurysm-related columns found in the training data"
        
        print("\n=== Training Data Statistics ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Columns: {stats['columns']}")
        
        if 'aneurysm_columns' in stats:
            print(f"Aneurysm distribution: {stats['aneurysm_distribution']}")
        else:
            print("No aneurysm-related columns found in the data")
            
        print("\nMissing values per column:")
        for col, count in stats['missing_values'].items():
            print(f"  - {col}: {count} missing")
        
        return stats
    
    def get_dicom_metadata(self, study_uid: str, series_uid: str) -> Dict:
        """Extract DICOM metadata for a given study and series UID."""
        series_path = self.series_dir / study_uid / f"{series_uid}.dcm"
        if not series_path.exists():
            return {}
            
        try:
            dicom_data = pydicom.dcmread(series_path, stop_before_pixels=True)
            return {
                "PatientID": getattr(dicom_data, "PatientID", ""),
                "Modality": getattr(dicom_data, "Modality", ""),
                "StudyDescription": getattr(dicom_data, "StudyDescription", ""),
                "SeriesDescription": getattr(dicom_data, "SeriesDescription", ""),
                "Rows": getattr(dicom_data, "Rows", 0),
                "Columns": getattr(dicom_data, "Columns", 0),
                "SliceThickness": getattr(dicom_data, "SliceThickness", 0),
                "PixelSpacing": getattr(dicom_data, "PixelSpacing", [0, 0]),
            }
        except Exception as e:
            print(f"Error reading DICOM {series_path}: {str(e)}")
            return {}
    
    def process_sample(self, idx: int) -> Dict:
        """Process a single sample from the training data."""
        if idx >= len(self.train_df):
            raise IndexError(f"Index {idx} out of range for training data")
            
        sample = self.train_df.iloc[idx].to_dict()
        
        # Check if we have the necessary UIDs
        series_uid = sample.get("SeriesInstanceUID")
        study_uid = sample.get("StudyInstanceUID", series_uid)  # Fallback to series_uid if study_uid not available
        
        # Get DICOM metadata if we have both UIDs
        dicom_meta = {}
        if study_uid and series_uid:
            dicom_meta = self.get_dicom_metadata(study_uid, series_uid)
        
        # Check for segmentation if we have both UIDs
        has_segmentation = False
        if study_uid and series_uid:
            seg_path = self.segmentations_dir / f"{study_uid}_{series_uid}.nii.gz"
            has_segmentation = seg_path.exists()
        
        return {
            **sample,
            "has_segmentation": has_segmentation,
            "StudyInstanceUID": study_uid,  # Ensure this key exists in the output
            "SeriesInstanceUID": series_uid,  # Ensure this key exists in the output
            **dicom_meta
        }
        
    def plot_aneurysm_distribution(self, save_path: str = None) -> None:
        """
        Plot the distribution of aneurysms by location.
        
        Args:
            save_path: Optional path to save the plot. If None, the plot is displayed.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get all location columns (excluding non-location columns)
        location_columns = [
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
        
        # Count positive cases per location
        location_counts = self.train_df[location_columns].sum().sort_values(ascending=False)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=location_counts.values, y=location_counts.index, palette='viridis')
        
        # Add count and percentage labels
        for i, v in enumerate(location_counts):
            percentage = (v / len(self.train_df)) * 100
            ax.text(v + 5, i, f"{v} ({percentage:.1f}%)", color='black', va='center')
        
        plt.title('Distribution of Aneurysms by Location', fontsize=14)
        plt.xlabel('Number of Cases', fontsize=12)
        plt.ylabel('Location', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    try:
        # Initialize processor
        processor = RSNADataProcessor()
        
        # Explore data
        stats = processor.explore_data()
        
        # Process first few samples as an example
        print("\n=== Processing first 3 samples ===")
        for i in range(min(3, len(processor.train_df))):
            sample = processor.process_sample(i)
            print(f"\nSample {i}:")
            print(f"  Patient: {sample.get('PatientID', 'N/A')}")
            print(f"  Modality: {sample.get('Modality', 'N/A')}")
            # Check for any aneurysm-related columns
            aneurysm_cols = [k for k in sample.keys() if 'aneurysm' in k.lower()]
            if aneurysm_cols:
                for col in aneurysm_cols:
                    print(f"  {col}: {sample.get(col, 'N/A')}")
            else:
                print("  No aneurysm-related data found in sample")
            print(f"  Has segmentation: {sample.get('has_segmentation', 'N/A')}")
            print(f"  Available keys: {list(sample.keys())}")
        
        # TODO: Add more processing steps here
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    main()
