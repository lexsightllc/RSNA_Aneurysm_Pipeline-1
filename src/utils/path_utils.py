"""
Path utilities for handling imports and file paths in both local and Kaggle environments.
"""
import os
import sys
from pathlib import Path

import os
import sys
from pathlib import Path
from typing import Dict, Any

def get_project_root() -> Path:
    """Get the project root directory, handling both local and Kaggle environments."""
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        # In Kaggle, the code is in /kaggle/input/rsna-aneurysm-pipeline-1
        return Path('/kaggle/input/rsna-aneurysm-pipeline-1')
    
    # For local development, go up from src/utils to get to project root
    return Path(__file__).parent.parent.parent

def setup_paths() -> Dict[str, Any]:
    """
    Set up Python path and return important directory paths.
    
    Returns:
        dict: Dictionary containing important paths and environment info
    """
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    project_root = get_project_root()
    
    # Add project root to Python path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # Define important directories
    paths = {
        'project_root': project_root,
        'src_dir': project_root / 'src',
        'config_dir': project_root / 'config',
        'data_dir': Path('/kaggle/input/rsna-intracranial-aneurysm-detection') if is_kaggle else project_root / 'data',
        'models_dir': project_root / 'models',
        'output_dir': Path('/kaggle/working') if is_kaggle else project_root / 'output',
        'is_kaggle': is_kaggle,
        'is_colab': 'COLAB_GPU' in os.environ,
        'is_local': not is_kaggle and 'COLAB_GPU' not in os.environ
    }
    
    # Create necessary directories in local environment
    if not is_kaggle:
        (project_root / 'data').mkdir(parents=True, exist_ok=True)
        (project_root / 'output').mkdir(parents=True, exist_ok=True)
        (project_root / 'models').mkdir(parents=True, exist_ok=True)
    
    return paths

# Initialize paths when module is imported
PATHS = setup_paths()

# Add commonly used paths as module-level variables
PROJECT_ROOT = PATHS['project_root']
SRC_DIR = PATHS['src_dir']
CONFIG_DIR = PATHS['config_dir']
DATA_DIR = PATHS['data_dir']
MODELS_DIR = PATHS['models_dir']
OUTPUT_DIR = PATHS['output_dir']
IS_KAGGLE = PATHS['is_kaggle']
IS_COLAB = PATHS['is_colab']
IS_LOCAL = PATHS['is_local']

def get_config_path(config_name: str) -> Path:
    """Get the full path to a config file."""
    return CONFIG_DIR / config_name

def get_model_path(model_name: str) -> Path:
    """Get the full path to a model file."""
    return MODELS_DIR / model_name
