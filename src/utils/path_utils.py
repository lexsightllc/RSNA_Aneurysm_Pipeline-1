"""
Path utilities for handling imports and file paths in both local and Kaggle environments.
"""
import os
import sys
from pathlib import Path

def setup_paths():
    """
    Set up Python path and return important directory paths.
    
    Returns:
        dict: Dictionary containing important paths
    """
    # Check if running in Kaggle environment
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    # Get the project root directory
    if is_kaggle:
        # In Kaggle, the notebook is in /kaggle/working
        project_root = Path('/kaggle/input/rsna-aneurysm-pipeline-1')
    else:
        # Local development - go up from src/utils to get to project root
        project_root = Path(__file__).parent.parent.parent
    
    # Add project root to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Define important directories
    paths = {
        'project_root': project_root,
        'src_dir': project_root / 'src',
        'config_dir': project_root / 'config',
        'data_dir': Path('/kaggle/input/rsna-intracranial-aneurysm-detection') if is_kaggle else project_root / 'data',
        'output_dir': Path('/kaggle/working') if is_kaggle else project_root / 'output',
        'is_kaggle': is_kaggle
    }
    
    # Create necessary directories
    if not is_kaggle:
        paths['data_dir'].mkdir(parents=True, exist_ok=True)
        paths['output_dir'].mkdir(parents=True, exist_ok=True)
    
    return paths

# Call setup_paths when this module is imported
PATHS = setup_paths()
