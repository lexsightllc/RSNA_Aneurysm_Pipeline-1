"""
RSNA Aneurysm Detection - Data Catalog

This module provides easy access to standardized data files with schema validation.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import polars as pl
from jsonschema import validate

# Constants
import os

# Check if running in Kaggle environment
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_KAGGLE:
    # Kaggle paths
    DATA_DIR = Path("/kaggle/input/rsna-intracranial-aneurysm-detection")
    OUTPUT_DIR = Path("/kaggle/working")
    SCHEMA_DIR = Path("/kaggle/input/rsna-aneurysm-pipeline-1/schemas")
    STANDARDIZED_DIR = OUTPUT_DIR / "standardized"
else:
    # Local development paths
    REPO_ROOT = Path(__file__).parent.parent
    DATA_DIR = REPO_ROOT / "data"
    STANDARDIZED_DIR = DATA_DIR / "standardized"
    SCHEMA_DIR = REPO_ROOT / "schemas"
    OUTPUT_DIR = REPO_ROOT / "output"

# Create necessary directories
STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load schemas
SCHEMAS = {}
for schema_file in SCHEMA_DIR.glob("*_schema.json"):
    with open(schema_file, 'r') as f:
        # Remove '_schema' from the key to match test expectations
        key = schema_file.stem.replace('_schema', '')
        SCHEMAS[key] = json.load(f)

class DataCatalog:
    """Main class for accessing standardized data files."""
    
    @staticmethod
    def load_predictions(file_path: Union[str, Path]) -> pl.DataFrame:
        """
        Load a predictions CSV file with validation.
        
        Args:
            file_path: Path to the predictions CSV file
            
        Returns:
            polars.DataFrame: DataFrame with validated predictions
            
        Raises:
            ValueError: If the data does not match the expected schema
        """
        # Load the schema once
        schema = SCHEMAS['rsna_labels']
        required_columns = set(schema.get('required', []))
        
        # Read the CSV file
        df = pl.read_csv(file_path)
        
        # Check for missing required columns
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
        
        # Validate each column based on the schema
        properties = schema.get('properties', {})
        for col_name, col_schema in properties.items():
            if col_name not in df.columns:
                continue  # Skip if column is optional and not present
                
            col_type = col_schema.get('type')
            min_val = col_schema.get('minimum')
            max_val = col_schema.get('maximum')
            
            # Check numeric constraints
            if col_type == 'number':
                col_min = df[col_name].min()
                col_max = df[col_name].max()
                if min_val is not None and col_min < min_val:
                    raise ValueError(f"Column '{col_name}' contains values below minimum {min_val} (min={col_min})")
                if max_val is not None and col_max > max_val:
                    raise ValueError(f"Column '{col_name}' contains values above maximum {max_val} (max={col_max})")
            
            # Add more type validations as needed (string, boolean, etc.)
            
        return df
    
    @staticmethod
    def load_calibration(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a calibration JSON file with validation.
        
        Args:
            file_path: Path to the calibration JSON file
            
        Returns:
            dict: Validated calibration data
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        validate(instance=data, schema=SCHEMAS['calibration'])
        return data
    
    @staticmethod
    def load_location_facts(file_path: Union[str, Path]) -> pl.DataFrame:
        """
        Load location facts from a JSONL file with validation.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            polars.DataFrame: DataFrame with validated location facts
        """
        with open(file_path, 'r') as f:
            items = [json.loads(line) for line in f]
        
        # Validate each item
        for item in items:
            validate(instance=item, schema=SCHEMAS['location_facts'])
        
        return pl.DataFrame(items)
    
    @staticmethod
    def load_standardized(file_path: Union[str, Path]) -> Any:
        """
        Load a standardized file with appropriate validation.
        
        Args:
            file_path: Path to the standardized file
            
        Returns:
            The loaded data (DataFrame, dict, or list) with validation
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            # Try to find in standardized directory
            file_path = STANDARDIZED_DIR / file_path
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            return DataCatalog.load_predictions(file_path)
        elif file_path.suffix.lower() == '.json':
            return DataCatalog.load_calibration(file_path)
        elif file_path.suffix.lower() == '.jsonl':
            return DataCatalog.load_location_facts(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    @classmethod
    def get_available_files(cls, pattern: str = "*") -> List[Path]:
        """
        List available standardized files matching a pattern.
        
        Args:
            pattern: Glob pattern to match filenames against
            
        Returns:
            List of Path objects to matching files
        """
        return list(STANDARDIZED_DIR.rglob(pattern))

# Example usage
if __name__ == "__main__":
    # Example: Load a predictions file
    # df = DataCatalog.load_standardized("path/to/predictions.csv")
    
    # Example: Load a calibration file
    # calib = DataCatalog.load_standardized("path/to/calibration.json")
    
    # Example: List available files
    print("Available standardized files:")
    for file_path in DataCatalog.get_available_files():
        print(f"- {file_path.relative_to(STANDARDIZED_DIR)}")
    
    print("\nUse DataCatalog.load_standardized() to load these files with validation.")
