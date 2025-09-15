"""
Tests for RSNA Aneurysm data schemas and validation.
"""

import os
import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from jsonschema import validate, ValidationError

# Add src directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import the schemas
SCHEMA_DIR = Path(__file__).parent.parent / "schemas"
SCHEMAS = {}
for schema_file in SCHEMA_DIR.glob("*_schema.json"):
    with open(schema_file, 'r') as f:
        # Remove '_schema' from the key to match test expectations
        key = schema_file.stem.replace('_schema', '')
        SCHEMAS[key] = json.load(f)

class TestRSNADictionaries(unittest.TestCase):
    """Test the RSNA label schemas and dictionaries."""
    
    def test_rsna_labels_schema(self):
        """Test that the RSNA labels schema is valid and complete."""
        schema = SCHEMAS['rsna_labels']
        
        # Should have all required fields
        required = schema['required']
        self.assertEqual(len(required), 14)
        
        # Check that all required fields are in properties
        for field in required:
            self.assertIn(field, schema['properties'])
        
        # Check that all properties are in required
        self.assertEqual(set(required), set(schema['properties'].keys()))
        
        # Check that no additional properties are allowed
        self.assertFalse(schema.get('additionalProperties', True))


class TestCalibrationSchema(unittest.TestCase):
    """Test the calibration schema."""
    
    def test_valid_calibration(self):
        """Test a valid calibration object."""
        schema = SCHEMAS['calibration']
        
        valid = {
            "x": [0.0, 0.5, 1.0],
            "y": [0.1, 0.4, 0.9],
            "brier": 0.15,
            "n_pos": 500,
            "label": "Aneurysm Present",
            "date_created": "2023-01-01T00:00:00Z",
            "model_version": "1.0.0"
        }
        
        # Should not raise an exception
        validate(instance=valid, schema=schema)
    
    def test_invalid_calibration(self):
        """Test an invalid calibration object."""
        schema = SCHEMAS['calibration']
        
        # Missing required fields
        invalid = {"x": [0, 1], "y": [0, 1, 2]}  # Different lengths
        with self.assertRaises(ValidationError):
            validate(instance=invalid, schema=schema)
        
        # Values out of range
        invalid = {"x": [0, 2], "y": [0, 1]}  # x > 1.0
        with self.assertRaises(ValidationError):
            validate(instance=invalid, schema=schema)


class TestLocationFactsSchema(unittest.TestCase):
    """Test the location facts schema."""
    
    def test_valid_location_facts(self):
        """Test a valid location facts object."""
        schema = SCHEMAS['location_facts']
        
        valid = {
            "rsna_label": "Left Middle Cerebral Artery",
            "prevalence_pct": 15.5,
            "bleed_pattern": "Sylvian fissure, temporal lobe",
            "clinical_signs": ["Contralateral hemiparesis", "Aphasia (if dominant hemisphere)"],
            "imaging_tips": ["Thin-slice CTA", "MRA TOF", "DSA for treatment planning"],
            "notes": "Most common site for middle cerebral artery aneurysms",
            "aliases": ["Left MCA", "LMCA"],
            "vessel_diameter_mm": {"min": 1.5, "max": 3.0, "mean": 2.4},
            "rupture_risk_factors": ["Size >7mm", "Irregular shape", "Prior SAH"],
            "treatment_considerations": ["Endovascular coiling", "Surgical clipping"],
            "references": ["PMID:12345678", "PMID:23456789"],
            "last_updated": "2023-01-01"
        }
        
        # Should not raise an exception
        validate(instance=valid, schema=schema)
    
    def test_minimal_location_facts(self):
        """Test a minimal valid location facts object."""
        schema = SCHEMAS['location_facts']
        
        minimal = {
            "rsna_label": "Right Posterior Communicating Artery"
        }
        
        # Should not raise an exception
        validate(instance=minimal, schema=schema)


class TestDataCatalog(unittest.TestCase):
    """Test the DataCatalog class."""
    
    def setUp(self):
        from src.data_catalog import DataCatalog
        self.catalog = DataCatalog
        
        # Create a temporary directory for test files
        self.test_dir = Path(__file__).parent / "test_data"
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a test predictions file with all required fields
        self.predictions_file = self.test_dir / "test_predictions.csv"
        
        # Define all required fields based on the schema
        required_fields = [
            "SeriesInstanceUID",
            "Left Infraclinoid Internal Carotid Artery",
            "Right Infraclinoid Internal Carotid Artery",
            "Left Supraclinoid Internal Carotid Artery",
            "Right Supraclinoid Internal Carotid Artery",
            "Left Middle Cerebral Artery",
            "Right Middle Cerebral Artery",
            "Anterior Communicating Artery",
            "Left Anterior Cerebral Artery",
            "Right Anterior Cerebral Artery",
            "Left Posterior Communicating Artery",
            "Right Posterior Communicating Artery",
            "Basilar Tip",
            "Other Posterior Circulation",
            "Aneurysm Present"
        ]
        
        # Create test data with all required fields
        # Note: The schema only expects the probability fields, not SeriesInstanceUID
        data = {
            "Left Middle Cerebral Artery": [0.1, 0.2],
            "Right Middle Cerebral Artery": [0.05, 0.15]
        }
        
        # Add all other required fields with default values
        for field in required_fields:
            if field not in data and field != "SeriesInstanceUID":
                data[field] = [0.01, 0.01]
                
        df = pd.DataFrame(data)
        df.to_csv(self.predictions_file, index=False)
    
    def tearDown(self):
        # Clean up test files
        if self.predictions_file.exists():
            self.predictions_file.unlink()
    
    def test_load_predictions(self):
        """Test loading predictions from a CSV file."""
        # Load the predictions
        df = self.catalog.load_predictions(self.predictions_file)
        
        # Should have 2 rows as per our test data
        self.assertEqual(len(df), 2)
        
        # Verify all required probability fields are present
        required_fields = [
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
        
        # Check all required fields are present and have correct type
        for field in required_fields:
            self.assertIn(field, df.columns)
            # Verify the values are numeric and within expected range
            self.assertTrue(all(0 <= val <= 1 for val in df[field]))
    
    def test_invalid_predictions(self):
        """Test loading invalid predictions."""
        # Create an invalid predictions file (missing required columns)
        invalid_file = self.test_dir / "invalid_predictions.csv"
        df = pd.DataFrame({
            "SeriesInstanceUID": ["1.2.3.4"],
            "Some Column": [0.5]
        })
        df.to_csv(invalid_file, index=False)
        
        with self.assertRaises(ValueError):
            self.catalog.load_predictions(invalid_file)
        
        # Clean up
        invalid_file.unlink()


if __name__ == "__main__":
    unittest.main()
