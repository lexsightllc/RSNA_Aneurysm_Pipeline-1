#!/usr/bin/env python3
"""
Legacy wrapper for the RC2 Offline Inference Runner.

This script is maintained for backward compatibility. For new code, use:
    python -m src.api.offline_runner --help
"""
import os
import sys
import warnings
from pathlib import Path

# Show deprecation warning
warnings.warn(
    "exec_rc2_offline.py is deprecated. Use 'python -m src.api.offline_runner' instead.",
    DeprecationWarning,
    stacklevel=2
)

def main():
    """Legacy entry point that forwards to the new implementation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Legacy wrapper for RC2 Offline Inference Runner"
    )
    parser.add_argument(
        "--series-root",
        required=True,
        help="Path to directory containing DICOM series subdirectories"
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output CSV file path (default: submission.csv)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of series to process (0 for no limit)"
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable calibration"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    # Parse known args to handle --help without failing on unknown args
    args, _ = parser.parse_known_args()
    
    # Import the main function from the new module
    try:
        from src.api.offline_runner import main as offline_runner_main
    except ImportError:
        # Fallback for development or direct execution
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.api.offline_runner import main as offline_runner_main
    
    # Map legacy args to new ones
    sys.argv = [
        sys.argv[0],
        "--series-root", args.series_root,
        "--output", args.out,
    ]
    
    if args.limit > 0:
        sys.argv.extend(["--limit", str(args.limit)])
    if args.no_calibration:
        sys.argv.append("--no-calibration")
    if args.debug:
        sys.argv.append("--debug")
    
    # Call the new implementation
    offline_runner_main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
