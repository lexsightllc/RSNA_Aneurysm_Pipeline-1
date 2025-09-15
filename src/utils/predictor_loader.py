"""
Centralized predictor loading for RSNA Aneurysm Detection.

This module provides a unified interface for loading different predictor implementations
with proper fallback mechanisms.
"""
import os
import sys
from typing import Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_predictor_adapter() -> Tuple[Any, str]:
    """
    Load the best available predictor adapter.
    
    Priority:
    1. User-provided rsna_aneurysm.AneurysmPredictor
    2. Bundled RC2 API stub (src.api.inference_server_rc2.predict)
    3. Deterministic prior-based stub
    
    Returns:
        Tuple of (predict_function, predictor_name)
    """
    # Try to import user's predictor first
    try:
        from rsna_aneurysm import AneurysmPredictor
        predictor = AneurysmPredictor()
        logger.info("Using custom AneurysmPredictor")
        return predictor.predict_series, "custom_predictor"
    except ImportError:
        logger.debug("Custom AneurysmPredictor not found, falling back to RC2 stub")
    
    # Fall back to bundled RC2 predictor
    try:
        from src.api.inference_server_rc2 import predict as rc2_predict
        logger.info("Using bundled RC2 predictor")
        return rc2_predict, "rc2_stub"
    except ImportError as e:
        logger.warning(f"Failed to load RC2 predictor: {e}")
    
    # Fall back to deterministic priors
    logger.warning("No predictor found, using deterministic priors")
    
    def predict_prior(series_path: str) -> dict:
        """Generate deterministic priors for testing."""
        import numpy as np
        from pathlib import Path
        from src.constants import RSNA_LOCATION_LABELS, RSNA_ANEURYSM_PRESENT_LABEL
        
        # Generate stable random values based on series path
        rng = np.random.RandomState(hash(Path(series_path).name) % 2**32)
        
        # Generate predictions slightly biased towards 0
        preds = {
            label: float(rng.beta(1.1, 10))  # Beta distribution with mean ~0.1
            for label in RSNA_LOCATION_LABELS
        }
        
        # Aneurysm present is max of location probs, slightly adjusted
        preds[RSNA_ANEURYSM_PRESENT_LABEL] = min(
            0.95, max(preds.values()) * 1.1
        )
        return preds
    
    return predict_prior, "deterministic_prior"

def setup_environment() -> None:
    """Set up environment for deterministic execution."""
    # Set environment variables for deterministic behavior
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Configure logging
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
