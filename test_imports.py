#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly.
Run this script to check if the environment is set up properly.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    print("Testing imports...")
    
    # Test basic imports
    import numpy as np
    import pandas as pd
    from src.utils.path_utils import PATHS
    from src.api.offline_runner import main as offline_runner_main
    from src.train_model import main as train_model_main
    
    print("All imports successful!")
    print("\nEnvironment paths:")
    print(f"Project root: {PATHS['project_root']}")
    print(f"Data directory: {PATHS['data_dir']}")
    print(f"Output directory: {PATHS['output_dir']}")
    
except ImportError as e:
    print(f"\nError importing modules: {e}", file=sys.stderr)
    print("\nPlease make sure you have installed all required dependencies:")
    print("pip install -r requirements.txt")
    print("\nAlso, ensure you run this script from the project root directory.")
    sys.exit(1)
