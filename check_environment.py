#!/usr/bin/env python3
"""
Environment and path verification script for RSNA Aneurysm Pipeline.
This script checks if all required directories exist and have the correct permissions.
"""
import os
import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import path utilities
from src.utils.path_utils import PATHS

def check_directory(path: Path, name: str, required: bool = True) -> bool:
    """Check if a directory exists and is accessible."""
    if not path.exists():
        if required:
            print(f"❌ {name} directory not found: {path}")
            return False
        else:
            print(f"ℹ️  {name} directory not found (optional): {path}")
            return True
    
    if not path.is_dir():
        print(f"❌ {name} path exists but is not a directory: {path}")
        return False
    
    if not os.access(path, os.R_OK):
        print(f"❌ No read permission for {name} directory: {path}")
        return False
    
    print(f"✅ {name} directory: {path}")
    return True

def check_file(path: Path, name: str, required: bool = True) -> bool:
    """Check if a file exists and is accessible."""
    if not path.exists():
        if required:
            print(f"❌ {name} file not found: {path}")
            return False
        else:
            print(f"ℹ️  {name} file not found (optional): {path}")
            return True
    
    if not path.is_file():
        print(f"❌ {name} path exists but is not a file: {path}")
        return False
    
    if not os.access(path, os.R_OK):
        print(f"❌ No read permission for {name} file: {path}")
        return False
    
    print(f"✅ {name} file: {path}")
    return True

def check_python_imports() -> bool:
    """Check if all required Python packages are installed."""
    print("\n🔍 Checking Python imports...")
    success = True
    
    try:
        import numpy
        import pandas
        import torch
        import torchvision
        import albumentations
        import pydicom
        import nibabel
        import skimage
        import matplotlib
        import sklearn
        import fastapi
        import uvicorn
        import yaml
        import jsonschema
        
        print("✅ All required Python packages are installed")
    except ImportError as e:
        print(f"❌ Missing Python package: {e.name}")
        success = False
    
    return success

def load_config() -> dict:
    """Load and validate the pipeline configuration."""
    config_path = PATHS['config_dir'] / 'pipeline_config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load config file: {e}")
        return {}

def main():
    print("🔧 RSNA Aneurysm Pipeline - Environment Check 🔧\n")
    
    # Check environment
    print("🌍 Environment:")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {PATHS['project_root']}")
    print(f"Running in {'Kaggle' if PATHS['is_kaggle'] else 'local'} environment")
    
    # Check directories
    print("\n📁 Checking directories:")
    dirs_ok = True
    dirs_ok &= check_directory(PATHS['project_root'], "Project root")
    dirs_ok &= check_directory(PATHS['src_dir'], "Source")
    dirs_ok &= check_directory(PATHS['config_dir'], "Config")
    dirs_ok &= check_directory(PATHS['data_dir'], "Data", required=False)
    dirs_ok &= check_directory(PATHS['output_dir'], "Output", required=False)
    
    # Check config files
    print("\n📄 Checking configuration files:")
    configs_ok = True
    configs_ok &= check_file(PATHS['config_dir'] / 'pipeline_config.json', "Pipeline config")
    configs_ok &= check_file(PATHS['config_dir'] / 'heuristics.yaml', "Heuristics config", required=False)
    configs_ok &= check_file(PATHS['config_dir'] / 'rsna_location.json', "RSNA location config", required=False)
    
    # Check Python imports
    imports_ok = check_python_imports()
    
    # Load and validate config
    config = load_config()
    config_ok = bool(config)
    
    # Summary
    print("\n📊 Summary:")
    print(f"Directories: {'✅' if dirs_ok else '❌'}")
    print(f"Config files: {'✅' if configs_ok else '❌'}")
    print(f"Python imports: {'✅' if imports_ok else '❌'}")
    print(f"Config validation: {'✅' if config_ok else '❌'}")
    
    if not all([dirs_ok, configs_ok, imports_ok, config_ok]):
        print("\n❌ Some checks failed. Please fix the issues above before proceeding.")
        sys.exit(1)
    
    print("\n✨ All checks passed! Your environment is ready to use. ✨")

if __name__ == "__main__":
    main()
