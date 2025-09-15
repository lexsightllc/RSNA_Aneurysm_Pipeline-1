#!/usr/bin/env python3
"""
Test script to verify the RSNA Aneurysm Pipeline is working correctly.
"""
import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required imports work."""
    logger.info("Testing imports...")
    try:
        import torch
        import numpy as np
        import pandas as pd
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import pydicom
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        logger.info("All imports successful!")
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False

def test_paths():
    """Test that all paths are set up correctly."""
    logger.info("Testing paths...")
    try:
        from src.utils.path_utils import PATHS, PROJECT_ROOT, SRC_DIR, CONFIG_DIR, DATA_DIR, OUTPUT_DIR
        
        paths = [
            ("Project Root", PROJECT_ROOT),
            ("Source Directory", SRC_DIR),
            ("Config Directory", CONFIG_DIR),
            ("Data Directory", DATA_DIR),
            ("Output Directory", OUTPUT_DIR)
        ]
        
        all_paths_ok = True
        for name, path in paths:
            path = Path(path)
            exists = path.exists()
            logger.info(f"{name}: {path} - {'✓' if exists else '✗'}")
            if not exists and name != "Data Directory":  # Data dir might not exist yet
                all_paths_ok = False
                
        return all_paths_ok
    except Exception as e:
        logger.error(f"Error testing paths: {e}")
        return False

def test_configs():
    """Test that configuration files exist and are valid."""
    logger.info("Testing configuration files...")
    try:
        from src.utils.path_utils import CONFIG_DIR
        import yaml
        import json
        
        config_files = [
            ("pipeline_config.json", json.load),
            ("heuristics.yaml", yaml.safe_load),
            ("rsna_location.json", json.load)
        ]
        
        all_configs_ok = True
        for filename, loader in config_files:
            try:
                path = CONFIG_DIR / filename
                with open(path, 'r') as f:
                    data = loader(f)
                logger.info(f"✓ {filename} is valid")
            except Exception as e:
                logger.error(f"✗ {filename} is invalid: {e}")
                all_configs_ok = False
                
        return all_configs_ok
    except Exception as e:
        logger.error(f"Error testing configs: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting RSNA Aneurysm Pipeline tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Path Test", test_paths),
        ("Config Test", test_configs)
    ]
    
    all_passed = True
    for name, test_func in tests:
        logger.info(f"\n--- {name} ---")
        try:
            if test_func():
                logger.info(f"✓ {name} PASSED")
            else:
                logger.error(f"✗ {name} FAILED")
                all_passed = False
        except Exception as e:
            logger.error(f"✗ {name} ERROR: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All tests passed!")
        return 0
    else:
        logger.error("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
