import os
from pathlib import Path

def check_paths():
    """Check and list contents of Kaggle competition paths."""
    paths = [
        "/kaggle/input/rsna-intracranial-aneurysm-detection",
        "/kaggle/input/rsna-aneurysm-pipeline-1",
        "/kaggle/working/"
    ]
    
    for path in paths:
        print(f"\n{'='*50}")
        print(f"Checking path: {path}")
        print(f"{'='*50}")
        
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue
            
        if not os.path.isdir(path):
            print(f"Path is not a directory: {path}")
            continue
            
        try:
            # List first 10 items in the directory
            items = list(Path(path).iterdir())
            print(f"Found {len(items)} items in {path}")
            
            for i, item in enumerate(items[:10]):  # Show first 10 items
                print(f"  - {item.name}" + (" (dir)" if item.is_dir() else ""))
                
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more items")
                
        except Exception as e:
            print(f"Error accessing {path}: {str(e)}")

if __name__ == "__main__":
    check_paths()
