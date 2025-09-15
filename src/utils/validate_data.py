#!/usr/bin/env python3
"""
RSNA Aneurysm Detection - Data Validation Script

Validates and standardizes all data files in the repository against defined schemas.
"""

import json
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from jsonschema import validate, ValidationError
from standardize_filenames import to_snake_case

# Constants
# Go up 3 levels from current file to reach project root
REPO_ROOT = Path(__file__).parent.parent.parent
SCHEMA_DIR = REPO_ROOT / "schemas"
DATA_DIR = REPO_ROOT / "data"
STANDARDIZED_DIR = DATA_DIR / "standardized"
REPORTS_DIR = REPO_ROOT / "reports"

# Canonical RSNA labels in the exact required order
CANONICAL_LABELS = [
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

@dataclass
class ValidationResult:
    """Container for validation results."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

class DataValidator:
    """Main validator class for RSNA Aneurysm data."""

    def __init__(self):
        self.schemas = self._load_schemas()
        self.alias_map = {}
        self._load_alias_map()

    def _load_schemas(self) -> Dict[str, Dict]:
        """Load all JSON schemas from the schemas directory."""
        schemas = {}
        for schema_file in SCHEMA_DIR.glob("*.json"):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schemas[schema_file.stem] = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Error loading schema {schema_file}: {e}", file=sys.stderr)
        return schemas

    def _load_alias_map(self):
        """Load label aliases from the glossary_synonyms.json file."""
        alias_file_path = REPO_ROOT / "config" / "glossary_synonyms.json"
        try:
            with open(alias_file_path, 'r', encoding='utf-8') as f:
                self.alias_map = json.load(f)
                if not isinstance(self.alias_map, dict):
                    print(f"Warning: Expected dictionary in {alias_file_path}, got {type(self.alias_map).__name__}", file=sys.stderr)
                    self.alias_map = {}
        except FileNotFoundError:
            self.alias_map = {}
            print(f"Warning: Alias map file not found at {alias_file_path}. Using empty map.", file=sys.stderr)
        except json.JSONDecodeError as e:
            self.alias_map = {}
            print(f"Error loading alias map from {alias_file_path}: {e}. Using empty map.", file=sys.stderr)
        except Exception as e:
            self.alias_map = {}
            print(f"Unexpected error loading alias map from {alias_file_path}: {e}", file=sys.stderr)

    def validate_file(self, file_path: Path) -> ValidationResult:
        """Validate a single file based on its type and content."""
        result = ValidationResult()

        try:
            suffix = file_path.suffix.lower()
            if suffix == '.json':
                self._validate_json_file(file_path, result)
            elif suffix == '.csv':
                self._validate_csv_file(file_path, result)
            elif suffix == '.jsonl':
                self._validate_jsonl_file(file_path, result)
            else:
                result.warnings.append(f"Unsupported file type: {suffix}")
        except (IOError, OSError, json.JSONDecodeError, pd.errors.EmptyDataError) as e:
            result.valid = False
            result.errors.append(f"Error processing {file_path}: {str(e)}")

        return result

    def _validate_json_file(self, file_path: Path, result: ValidationResult):
        """Validate a JSON file against the appropriate schema."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if this is a calibration file
            if 'x' in data and 'y' in data:
                self._validate_against_schema(data, 'calibration', result)
            # Check if this is a location facts file
            elif 'rsna_label' in data:
                self._validate_against_schema(data, 'location_facts', result)
            # Check if this matches the RSNA labels structure
            elif all(label in data for label in CANONICAL_LABELS):
                self._validate_against_schema(data, 'rsna_labels', result)
            else:
                result.warnings.append("No matching schema found for JSON structure")

        except json.JSONDecodeError as e:
            result.valid = False
            result.errors.append(f"Invalid JSON: {str(e)}")

    def _validate_csv_file(self, file_path: Path, result: ValidationResult):
        """Validate a CSV file against the appropriate schema."""
        try:
            # Try to read with pandas first for better error messages
            df = pd.read_csv(file_path)

            # Check if this is a predictions/priors file
            if set(CANONICAL_LABELS).issubset(df.columns):
                self._validate_predictions(df, result)
            # Add more CSV validation as needed

            result.stats.update({
                'rows': str(len(df)),
                'columns': ', '.join(df.columns),
                'missing_pct': ', '.join(f"{k}: {v:.1%}" for k, v in df.isnull().mean().items())
            })

        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            result.valid = False
            result.errors.append(f"CSV parsing error: {str(e)}")
        except (IOError, OSError) as e:
            result.valid = False
            result.errors.append(f"File I/O error: {str(e)}")

    def _determine_schema_name(self, data: Any) -> Optional[str]:
        """Determine the schema name based on the data structure."""
        if not isinstance(data, dict):
            return None

        if 'x' in data and 'y' in data:
            return 'calibration'
        if 'rsna_label' in data:
            return 'location_facts'
        if all(label in data for label in CANONICAL_LABELS):
            return 'rsna_labels'
        return None

    def _process_jsonl_lines(
        self,
        lines: List[Any],
        schema_name: str,
        result: ValidationResult
    ) -> None:
        """Process and validate JSONL lines against the given schema."""
        for i, item in enumerate(lines, 1):
            if not isinstance(item, dict):
                result.warnings.append(
                    f"Line {i}: Expected JSON object, got {type(item).__name__}"
                )
                continue
            self._validate_against_schema(
                item, schema_name, result, f"Line {i}: "
            )

    def _read_jsonl_file(self, file_path: Path, result: ValidationResult) -> Optional[List[Any]]:
        """Read and parse a JSONL file, returning the parsed lines."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        result.warnings.append(f"Invalid JSON on line {i}: {str(e)}")
                return lines
        except (IOError, OSError) as e:
            result.valid = False
            result.errors.append(f"File I/O error: {str(e)}")
            return None

    def _validate_jsonl_file(self, file_path: Path, result: ValidationResult) -> None:
        """Validate a JSONL file."""
        try:
            # Read and parse the JSONL file
            lines = self._read_jsonl_file(file_path, result)
            if lines is None:  # Error occurred during file reading
                return

            # Basic validation
            if not lines:
                result.warnings.append("Empty JSONL file")
                return

            # Determine schema from first line
            first_line = lines[0]
            if isinstance(first_line, dict):
                schema_name = self._determine_schema_name(first_line)
                if schema_name:
                    self._process_jsonl_lines(lines, schema_name, result)

            # Update statistics
            result.stats.update({
                'line_count': str(len(lines)),
                'sample': str(first_line)[:100] + ('...' if len(str(first_line)) > 100 else '')
            })

        except Exception as e:  # pylint: disable=broad-except
            result.valid = False
            result.errors.append(f"Unexpected error processing JSONL: {str(e)}")

    def _check_required_columns(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for missing required columns in the DataFrame."""
        missing_cols = set(CANONICAL_LABELS) - set(df.columns)
        if missing_cols:
            result.valid = False
            result.errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    def _validate_probability_ranges(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate that all probability values are between 0 and 1."""
        for col in CANONICAL_LABELS:
            if col in df.columns and (df[col].min() < 0 or df[col].max() > 1):
                result.errors.append(f"Values in {col} must be between 0 and 1")

    def _validate_series_instance_uid(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for SeriesInstanceUID in multi-row files."""
        if 'SeriesInstanceUID' not in df.columns and len(df) > 1:
            result.warnings.append("No SeriesInstanceUID column found in multi-row file")

    def _validate_predictions(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate a DataFrame containing predictions or priors."""
        self._check_required_columns(df, result)
        self._validate_probability_ranges(df, result)
        self._validate_series_instance_uid(df, result)

    def _validate_against_schema(
        self,
        data: Dict,
        schema_name: str,
        result: ValidationResult,
        prefix: str = ""
    ) -> None:
        """Validate data against a specific schema."""
        if schema_name not in self.schemas:
            result.warnings.append(f"Schema {schema_name} not found")
            return

        try:
            validate(instance=data, schema=self.schemas[schema_name])
        except ValidationError as e:
            result.valid = False
            result.errors.append(f"{prefix}{str(e)}")

    def standardize_file(self, file_path: Path, output_dir: Path) -> Tuple[bool, str]:
        """Create a standardized version of a file.

        Args:
            file_path: Path to the input file
            output_dir: Directory to save the standardized file

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create standardized filename
            new_name = to_snake_case(file_path.stem) + file_path.suffix
            output_path = output_dir / new_name

            # For text files, we can clean them up
            if file_path.suffix.lower() in ('.csv', '.json', '.jsonl'):
                with open(output_path, 'w', encoding='utf-8') as f:
                    if file_path.suffix.lower() == '.csv':
                        # Read and rewrite CSV to standardize formatting
                        df = pd.read_csv(file_path)
                        df.to_csv(f, index=False)
                    elif file_path.suffix.lower() == '.json':
                        with open(file_path, 'r', encoding='utf-8') as src:
                            data = json.load(src)
                        json.dump(data, f, indent=2)
                    elif file_path.suffix.lower() == '.jsonl':
                        with open(file_path, 'r', encoding='utf-8') as src:
                            for line in src:
                                f.write(line)
            else:
                # For non-text files, just copy
                shutil.copy2(file_path, output_path)

            return True, str(output_path)

        except (IOError, OSError, json.JSONDecodeError, pd.errors.EmptyDataError) as e:
            return False, f"Error standardizing file {file_path}: {str(e)}"

def main() -> None:
    """Main entry point for the validation script."""
    parser = ArgumentParser(description='Validate RSNA Aneurysm data files')
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically fix issues when possible'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='validation_report.md',
        help='Output report file'
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Specific files to validate (default: all files)'
    )
    parser.add_argument(
        '--standardize',
        action='store_true',
        help='Create standardized versions of all files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=STANDARDIZED_DIR,
        help=f'Output directory for standardized files (default: {STANDARDIZED_DIR})'
    )
    args = parser.parse_args()

    # Ensure output directory exists
    if args.standardize:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize validator
    validator = DataValidator()

    # Find all data files
    data_files = []
    for ext in ('*.csv', '*.json', '*.jsonl'):
        data_files.extend(REPO_ROOT.rglob(ext))

    # Filter out files in ignored directories
    ignored_dirs = {'__pycache__', '.git', 'venv', 'env', '.ipynb_checkpoints'}
    data_files = [f for f in data_files if not any(part in ignored_dirs for part in f.parts)]

    # Process each file
    results = {}
    for file_path in sorted(data_files):
        rel_path = file_path.relative_to(REPO_ROOT)
        print(f"Validating {rel_path}...", end=' ', flush=True)

        # Validate
        result = validator.validate_file(file_path)
        results[str(rel_path)] = {
            'valid': result.valid,
            'errors': result.errors,
            'warnings': result.warnings,
            'stats': result.stats
        }

        # Standardize if requested
        if args.standardize and result.valid:
            success, output_path = validator.standardize_file(file_path, args.output_dir)
            if success:
                print(f"✓ (standardized to {output_path})")
            else:
                print(f"✗ (standardization failed: {output_path})")
        else:
            status = "✓" if result.valid else "✗"
            print(f"{status} ({len(result.errors)} errors, {len(result.warnings)} warnings)")

    # Generate reports
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / 'data_quality_report.md', 'w', encoding='utf-8') as f:
        _generate_markdown_report(results, f)

    with open(REPORTS_DIR / 'data_quality_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Exit with error code if any validation failed
    if any(not r['valid'] for r in results.values()):
        print("\nValidation failed with errors. See reports/ for details.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll data validated successfully!")

def _generate_markdown_report(results: Dict, output_file: Path) -> None:
    """Generate a markdown report from validation results."""
    output_file.write("# RSNA Aneurysm Data Quality Report\n\n")
    output_file.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

    # Summary
    valid_count = sum(1 for r in results.values() if r['valid'])
    total_count = len(results)
    error_count = sum(len(r['errors']) for r in results.values())
    warning_count = sum(len(r['warnings']) for r in results.values())

    output_file.write("## Summary\n\n")
    output_file.write(f"- ✅ **{valid_count}/{total_count}** files passed validation\n")
    output_file.write(f"- ❌ **{error_count}** errors found\n")
    output_file.write(f"- ⚠️  **{warning_count}** warnings found\n\n")

    # Detailed results
    output_file.write("## File Details\n\n")
    for file_path, result in results.items():
        status = "✅" if result['valid'] else "❌"
        output_file.write(f"### {status} `{file_path}`\n\n")

        if not result['valid']:
            output_file.write("#### Errors\n\n")
            for error in result['errors']:
                output_file.write(f"- ❌ {error}\n")
            output_file.write("\n")

        if result['warnings']:
            output_file.write("#### Warnings\n\n")
            for warning in result['warnings']:
                output_file.write(f"- ⚠️  {warning}\n")
            output_file.write("\n")

        if result['stats']:
            output_file.write("#### Statistics\n\n")
            if 'rows' in result['stats']:
                output_file.write(f"- Rows: {result['stats']['rows']:,}\n")
            if 'columns' in result['stats']:
                output_file.write(f"- Columns: {', '.join(result['stats']['columns'])}\n")
            output_file.write("\n")

    output_file.write("---\n*Report generated by RSNA Aneurysm Data Validator*\n")

if __name__ == "__main__":
    main()
