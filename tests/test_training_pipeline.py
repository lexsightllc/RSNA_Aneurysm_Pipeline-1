#!/usr/bin/env python3
"""
Test script for the RSNA Intracranial Aneurysm Detection training pipeline.

This script tests the training pipeline with a small subset of data.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import pytest
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.train_model import (
    AneurysmDataset,
    AneurysmTrainer,
    load_training_data,
    setup_logging,
    load_config,
)

# Test configuration
TEST_CONFIG = {
    'data_dir': 'data',  # Will be updated in setup
    'output_dir': None,  # Will be set to a temp directory
    'num_folds': 2,
    'batch_size': 2,
    'num_epochs': 1,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 2,
    'use_focal_loss': True,
    'focal_alpha': 1.0,
    'focal_gamma': 2.0,
    'gradient_clip': 1.0,
    'num_workers': 0,  # Set to 0 for testing to avoid multiprocessing issues
    'seed': 42,
    'debug': True,  # Enable debug mode for testing
}


def create_dummy_data(data_dir: Path, num_series: int = 4):
    """Create dummy data for testing."""
    # Create dummy DICOM series directories
    series_dirs = []
    for i in range(num_series):
        series_dir = data_dir / 'train' / f'study_{i}' / f'series_{i}'
        series_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy DICOM files (empty files for testing)
        for j in range(10):  # 10 slices per series
            (series_dir / f'image_{j:03d}.dcm').touch()
        
        series_dirs.append(str(series_dir.relative_to(data_dir)))
    
    # Create dummy labels
    labels = {
        'SeriesInstanceUID': [f'series_{i}' for i in range(num_series)],
        'Aneurysm Present': np.random.randint(0, 2, size=num_series).tolist(),
    }
    
    # Add location columns
    for loc in ['A1', 'A2', 'AcomA', 'BA', 'ICA', 'MCA', 'P1', 'P2', 'PcomA', 'VA', 'Other']:
        labels[loc] = np.random.randint(0, 2, size=num_series).tolist()
    
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(data_dir / 'train_labels.csv', index=False)
    
    return series_dirs, labels_df


def test_aneurysm_dataset():
    """Test the AneurysmDataset class."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        series_dirs, labels_df = create_dummy_data(temp_dir, num_series=4)
        
        # Create dataset
        dataset = AneurysmDataset(
            series_paths=series_dirs,
            labels=labels_df,
            target_size=(64, 64, 32)  # Smaller size for testing
        )
        
        # Test dataset length
        assert len(dataset) == 4
        
        # Test getting an item
        item = dataset[0]
        assert 'volume' in item
        assert 'labels' in item
        assert item['volume'].shape == (1, 64, 64, 32)  # (C, D, H, W)
        assert item['labels'].shape == (11,)  # 1 for presence + 10 locations


def test_trainer_initialization():
    """Test the AneurysmTrainer initialization."""
    config = TEST_CONFIG.copy()
    trainer = AneurysmTrainer(config)
    
    # Check if model is on the correct device
    assert next(trainer.model.parameters()).device.type in ('cuda', 'cpu')
    
    # Check optimizer and loss function
    assert isinstance(trainer.optimizer, optim.AdamW)
    assert isinstance(trainer.criterion, (torch.nn.BCEWithLogitsLoss, type(trainer.criterion)))  # FocalLoss or BCEWithLogitsLoss


def test_training_loop():
    """Test a single training loop iteration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        series_dirs, labels_df = create_dummy_data(temp_dir, num_series=4)
        
        # Update config
        config = TEST_CONFIG.copy()
        config['data_dir'] = str(temp_dir)
        config['output_dir'] = str(temp_dir / 'output')
        
        # Create trainer
        trainer = AneurysmTrainer(config)
        
        # Create a small dataset and dataloader
        dataset = AneurysmDataset(
            series_paths=series_dirs,
            labels=labels_df,
            target_size=(64, 64, 32)  # Smaller size for testing
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0  # Set to 0 for testing
        )
        
        # Test training for one epoch
        train_loss = trainer.train_epoch(dataloader)
        assert isinstance(train_loss, float)
        
        # Test validation
        val_loss, val_auc = trainer.validate_epoch(dataloader)
        assert isinstance(val_loss, float)
        assert 0 <= val_auc <= 1.0  # AUC should be between 0 and 1


def test_load_training_data():
    """Test the load_training_data function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        series_dirs, labels_df = create_dummy_data(temp_dir, num_series=4)
        
        # Test loading the dummy data
        series_paths, labels = load_training_data(str(temp_dir))
        
        # Check if the correct number of series were loaded
        assert len(series_paths) == 4
        assert len(labels) == 4
        
        # Check if all required columns are present
        required_columns = ['SeriesInstanceUID', 'Aneurysm Present'] + \
                          ['A1', 'A2', 'AcomA', 'BA', 'ICA', 'MCA', 'P1', 'P2', 'PcomA', 'VA', 'Other']
        
        for col in required_columns:
            assert col in labels.columns, f"Missing column: {col}"


if __name__ == "__main__":
    # Run tests
    test_aneurysm_dataset()
    test_trainer_initialization()
    test_training_loop()
    test_load_training_data()
    print("All tests passed!")
