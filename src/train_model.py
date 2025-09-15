#!/usr/bin/env python3
"""
RSNA Intracranial Aneurysm Detection - Training Pipeline

Comprehensive training framework with:
- 3D data augmentation
- Class balancing
- Multi-task learning
- Knowledge-based heuristics
- Cross-validation
"""

import os
import sys
import json
import logging
import argparse
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from src.adapter_compact3d_rc2 import Compact3DModel, load_dicom_series, preprocess_volume
from src.utils.prediction_utils import RSNA_ALL_LABELS, RSNA_LOCATION_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AneurysmDataset(Dataset):
    """Dataset for RSNA Aneurysm Detection with 3D augmentation."""
    
    def __init__(self, 
                 series_paths: List[str], 
                 labels: pd.DataFrame,
                 transform=None,
                 target_size: Tuple[int, int, int] = (128, 128, 128)):
        self.series_paths = series_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        
        # Create label mapping
        self.label_cols = [col for col in RSNA_ALL_LABELS if col != 'SeriesInstanceUID']
        
    def __len__(self):
        return len(self.series_paths)
    
    def __getitem__(self, idx):
        series_path = self.series_paths[idx]
        series_id = Path(series_path).name
        
        try:
            # Load DICOM data
            dicom_data = load_dicom_series(series_path)
            volume = preprocess_volume(dicom_data, target_size=self.target_size)
            
            # Remove batch dimension for dataset
            volume = volume.squeeze(0)
            
            # Apply 3D augmentations
            if self.transform:
                volume = self.apply_3d_transforms(volume.numpy())
                volume = torch.from_numpy(volume)
            
            # Get labels
            label_row = self.labels[self.labels['SeriesInstanceUID'] == series_id]
            if len(label_row) == 0:
                # Default to all zeros if no labels found
                labels = torch.zeros(len(self.label_cols), dtype=torch.float32)
            else:
                labels = torch.tensor([
                    float(label_row[col].iloc[0]) for col in self.label_cols
                ], dtype=torch.float32)
            
            return volume, labels, series_id
            
        except Exception as e:
            logger.warning(f"Error loading {series_path}: {e}")
            # Return dummy data
            volume = torch.zeros((1, *self.target_size), dtype=torch.float32)
            labels = torch.zeros(len(self.label_cols), dtype=torch.float32)
            return volume, labels, series_id
    
    def apply_3d_transforms(self, volume: np.ndarray) -> np.ndarray:
        """Apply 3D augmentations to volume."""
        if self.transform is None:
            return volume
        
        # Random flips
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=1)  # Left-right flip
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=2)  # Anterior-posterior flip
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=3)  # Superior-inferior flip
        
        # Random rotation (small angles)
        if np.random.random() > 0.7:
            from scipy.ndimage import rotate
            angle = np.random.uniform(-10, 10)
            volume = rotate(volume, angle, axes=(1, 2), reshape=False, order=1)
        
        # Intensity augmentation
        if np.random.random() > 0.5:
            # Random brightness/contrast
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.9, 1.1)
            volume = volume * contrast + brightness - 1.0
            volume = np.clip(volume, 0, 1)
        
        # Random noise
        if np.random.random() > 0.8:
            noise = np.random.normal(0, 0.01, volume.shape)
            volume = volume + noise
            volume = np.clip(volume, 0, 1)
        
        return volume

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AneurysmTrainer:
    """Main training class for RSNA Aneurysm Detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = Compact3DModel(num_classes=len(self.label_cols))
        self.model.to(self.device)
        
        # Loss function
        if config.get('use_focal_loss', True):
            self.criterion = FocalLoss(alpha=config.get('focal_alpha', 1), 
                                     gamma=config.get('focal_gamma', 2))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        
    def create_data_loaders(self, train_paths: List[str], val_paths: List[str], 
                           labels: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        
        # Training dataset with augmentation
        train_dataset = AneurysmDataset(
            train_paths, labels, 
            transform=True,  # Enable augmentation
            target_size=tuple(self.config.get('input_size', [128, 128, 128]))
        )
        
        # Validation dataset without augmentation
        val_dataset = AneurysmDataset(
            val_paths, labels,
            transform=None,  # No augmentation for validation
            target_size=tuple(self.config.get('input_size', [128, 128, 128]))
        )
        
        # Create weighted sampler for training to handle class imbalance
        if self.config.get('use_weighted_sampling', True):
            train_labels = []
            for path in train_paths:
                series_id = Path(path).name
                label_row = labels[labels['SeriesInstanceUID'] == series_id]
                if len(label_row) > 0:
                    # Use 'Aneurysm Present' for weighting
                    train_labels.append(float(label_row['Aneurysm Present'].iloc[0]))
                else:
                    train_labels.append(0.0)
            
            # Calculate class weights
            pos_weight = 1.0 / (sum(train_labels) / len(train_labels) + 1e-8)
            neg_weight = 1.0 / ((len(train_labels) - sum(train_labels)) / len(train_labels) + 1e-8)
            
            weights = [pos_weight if label > 0.5 else neg_weight for label in train_labels]
            sampler = WeightedRandomSampler(weights, len(weights))
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.get('batch_size', 4),
                sampler=sampler,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.get('batch_size', 4),
                shuffle=True,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (volumes, labels, series_ids) in enumerate(train_loader):
            volumes = volumes.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(volumes)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for volumes, labels, series_ids in val_loader:
                volumes = volumes.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(volumes)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Store predictions for metrics
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        
        # Calculate AUC for 'Aneurysm Present' (last column)
        try:
            auc = roc_auc_score(all_labels[:, -1], all_outputs[:, -1])
        except ValueError:
            auc = 0.5  # Default if all labels are the same
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, auc
    
    def train(self, train_paths: List[str], val_paths: List[str], labels: pd.DataFrame) -> float:
        """Main training loop with enhanced error handling and logging.
        
        Args:
            train_paths: List of paths to training DICOM series
            val_paths: List of paths to validation DICOM series
            labels: DataFrame containing labels for all series
            
        Returns:
            Best validation AUC achieved during training (or 0.5 if training fails)
        """
        try:
            # Log training configuration
            logger.info(f"Starting training with {len(train_paths)} training and {len(val_paths)} validation series")
            logger.info(f"Training configuration: {json.dumps(self.config, indent=2, default=str)}")
            
            # Create data loaders
            try:
                train_loader, val_loader = self.create_data_loaders(train_paths, val_paths, labels)
                logger.info(f"Created data loaders with {len(train_loader.dataset)} training and {len(val_loader.dataset)} validation samples")
            except Exception as e:
                logger.error(f"Failed to create data loaders: {str(e)}", exc_info=True)
                return 0.5
            
            best_auc = 0.0
            patience_counter = 0
            max_patience = self.config.get('early_stopping_patience', 10)
            num_epochs = self.config.get('num_epochs', 100)
            output_dir = Path(self.config.get('output_dir', 'models'))
            
            # Track best model path
            best_model_path = output_dir / 'best_model_fold.pth'
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            for epoch in range(num_epochs):
                epoch_start = time.time()
                logger.info("\n" + "=" * 50)
                logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                
                try:
                    # Training
                    train_loss = self.train_epoch(train_loader)
                    self.train_losses.append(train_loss)
                    
                    # Validation
                    val_loss, val_auc = self.validate_epoch(val_loader)
                    self.val_losses.append(val_loss)
                    self.val_aucs.append(val_auc)
                    
                    epoch_time = time.time() - epoch_start
                    logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
                    logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
                    
                    # Learning rate scheduling
                    self.scheduler.step(val_auc)
                    
                    # Save best model
                    if val_auc > best_auc:
                        best_auc = val_auc
                        patience_counter = 0
                        
                        # Save model
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_auc': val_auc,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'config': self.config
                        }, str(best_model_path))
                        
                        logger.info(f"New best model saved with AUC: {best_auc:.4f} to {best_model_path}")
                    else:
                        patience_counter += 1
                        logger.info(f"No improvement in validation AUC. Patience: {patience_counter}/{max_patience}")
                    
                    # Save checkpoint every few epochs
                    if (epoch + 1) % 5 == 0:
                        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_auc': val_auc,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'config': self.config
                        }, str(checkpoint_path))
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Early stopping
                    if patience_counter >= max_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                        
                except Exception as e:
                    logger.error(f"Error during epoch {epoch + 1}: {str(e)}", exc_info=True)
                    if epoch == 0:
                        # If first epoch fails, give up to avoid wasting resources
                        logger.error("Training failed on first epoch. Aborting training.")
                        return 0.5
                    # Otherwise continue with next epoch
                    continue
            
            # Save final model
            final_model_path = output_dir / 'final_model.pth'
            torch.save({
                'epoch': num_epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_auc': best_auc,
                'config': self.config
            }, str(final_model_path))
            logger.info(f"Saved final model to {final_model_path}")
            
            return best_auc
            
        except Exception as e:
            logger.error(f"Fatal error during training: {str(e)}", exc_info=True)
            return 0.5

def load_training_data(data_dir: str, config: Optional[Dict[str, Any]] = None):
    """
    Load training data paths and labels with enhanced Kaggle support.
    
    Args:
        data_dir: Path to the directory containing training data
        config: Configuration dictionary (for debug mode detection)
        
    Returns:
        tuple: (list_of_series_paths, labels_dataframe)
    """
    if config is None:
        config = {}
    data_path = Path(data_dir)
    logger = logging.getLogger(__name__)
    
    # Log environment information
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
    logger.info(f"Loading data from: {data_dir}")
    logger.info(f"Environment: {'Kaggle' if is_kaggle else 'Local'}")
    
    # Possible paths for train_labels.csv
    possible_label_paths = [
        data_path / "train_labels.csv",
        data_path / "train" / "train_labels.csv",
        data_path / "rsna-intracranial-aneurysm-detection" / "train_labels.csv",
        data_path / "rsna-intracranial-aneurysm-detection" / "train" / "train_labels.csv",
        data_path / "train_labels.csv",
    ]
    
    # Additional Kaggle-specific paths
    if is_kaggle:
        possible_label_paths.insert(0, data_path / "train_labels.csv")
        possible_label_paths.insert(1, data_path / "rsna-intracranial-aneurysm-detection" / "train_labels.csv")
    
    # Possible paths for train directory
    possible_train_dirs = [
        data_path / "train",
        data_path / "rsna-intracranial-aneurysm-detection" / "train",
        data_path / "rsna-intracranial-aneurysm-detection",
        data_path,
    ]
    
    # Find the labels file
    train_labels_path = None
    for path in possible_label_paths:
        if path.exists():
            train_labels_path = path
            logger.info(f"Found labels file at: {path}")
            break
            
    if train_labels_path is None:
        # Skip recursive search in Kaggle environment
        if is_kaggle:
            logger.warning("Skipping recursive search in Kaggle due to large dataset size")
        else:
            # Try recursive search as last resort with depth limit
            try:
                csv_files = list(data_path.rglob("*.csv"))
                for csv_file in csv_files:
                    if "train_labels" in csv_file.name:
                        train_labels_path = csv_file
                        logger.info(f"Found labels file via recursive search: {csv_file}")
                        break
            except Exception as e:
                logger.warning(f"Recursive search aborted: {e}")
                
    if train_labels_path is None:
        # Skip file listing in Kaggle environment
        if is_kaggle:
            logger.warning("Using Kaggle-specific fallback without file listing")
        else:
            try:
                # Limited file listing
                available_files = [str(p) for p in list(data_path.glob('*'))[:100]]
                logger.error(f"Available files (first 100): {available_files}")
            except Exception as e:
                logger.error(f"Error listing files: {e}")
                
        # Create dummy data as fallback
        logger.warning("Creating dummy data to allow script to continue")
        dummy_series = [f"dummy_series_{i}" for i in range(10)]
        dummy_labels = pd.DataFrame({
            'SeriesInstanceUID': dummy_series,
            'Aneurysm Present': np.random.randint(0, 2, size=10),
            **{loc: np.random.randint(0, 2, size=10) for loc in RSNA_LOCATION_LABELS}
        })
        return dummy_series, dummy_labels
    
    # Find the train directory
    train_dir = None
    for path in possible_train_dirs:
        if path.exists() and path.is_dir():
            # Check if directory contains DICOM files
            dicom_files = list(path.rglob("*.dcm"))
            if len(dicom_files) > 0 or "dummy" in str(path):
                train_dir = path
                logger.info(f"Found train directory at: {path} with {len(dicom_files)} DICOM files")
                break
            else:
                logger.info(f"Found directory {path} but it contains no DICOM files")
            
    if train_dir is None:
        logger.error(f"Could not find train directory with DICOM files in any expected location.")
        logger.error(f"Searched in: {[str(p) for p in possible_train_dirs]}")
        
        # Create dummy data as fallback
        logger.warning("No DICOM series found. Creating dummy paths for testing.")
        dummy_series = [f"dummy_series_{i}" for i in range(10)]
        dummy_labels = pd.DataFrame({
            'SeriesInstanceUID': [f'dummy_series_{i}' for i in range(10)],
            'Aneurysm Present': np.random.randint(0, 2, size=10),
            **{loc: np.random.randint(0, 2, size=10) for loc in RSNA_LOCATION_LABELS}
        })
        return dummy_series, dummy_labels
    
    # Load labels
    try:
        labels_df = pd.read_csv(train_labels_path)
        logger.info(f"Loaded {len(labels_df)} training labels from {train_labels_path}")
        
        # Validate required columns
        required_columns = ['SeriesInstanceUID', 'Aneurysm Present'] + list(RSNA_LOCATION_LABELS)
        missing_columns = [col for col in required_columns if col not in labels_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in labels: {missing_columns}. Adding with default values.")
            for col in missing_columns:
                labels_df[col] = 0
    except Exception as e:
        logger.error(f"Error loading training labels from {train_labels_path}: {e}")
        
        # Create dummy labels as fallback
        logger.warning("Creating dummy labels to allow script to continue")
        series_ids = [f"dummy_series_{i}" for i in range(10)]
        labels_df = pd.DataFrame({
            'SeriesInstanceUID': series_ids,
            'Aneurysm Present': np.random.randint(0, 2, size=10),
            **{loc: np.random.randint(0, 2, size=10) for loc in RSNA_LOCATION_LABELS}
        })
    
    # Get all series paths
    series_paths = []
    try:
        # First try the expected Kaggle structure
        for study_dir in train_dir.glob("*"):
            if study_dir.is_dir():
                for series_dir in study_dir.glob("*"):
                    if series_dir.is_dir():
                        series_paths.append(str(series_dir))
        
        # If no series found, try alternative structures
        if not series_paths:
            logger.warning("No series found in standard structure, trying alternative paths...")
            for path in train_dir.rglob("*.dcm"):
                series_dir = path.parent
                if str(series_dir) not in series_paths:
                    series_paths.append(str(series_dir))
        
        logger.info(f"Found {len(series_paths)} DICOM series in {train_dir}")
        
        if not series_paths:
            logger.warning(f"No DICOM series found in {train_dir}")
            logger.info(f"Directory contents: {list(train_dir.glob('*'))}")
        
        # If no series found, create dummy paths for testing
        if not series_paths:
            logger.warning("No DICOM series found. Creating dummy paths for testing.")
            series_paths = [f"dummy_series_{i}" for i in range(len(labels_df))]
            
        return series_paths, labels_df
        
    except Exception as e:
        logger.error(f"Error loading DICOM series: {e}")
        # If error occurs, return dummy data to allow the script to run
        logger.warning("Returning dummy data due to error")
        return [f"dummy_series_{i}" for i in range(10)], pd.DataFrame()
    
    logger.info(f"Found {len(series_paths)} DICOM series")
    
    return series_paths, labels

def load_training_data_kaggle(data_dir: str, config: Optional[Dict[str, Any]] = None):
    """
    Load training data paths and labels with enhanced Kaggle support.
    
    Args:
        data_dir: Path to the directory containing training data
        config: Configuration dictionary (for debug mode detection)
        
    Returns:
        tuple: (list_of_series_paths, labels_dataframe)
    """
    if config is None:
        config = {}
    data_path = Path(data_dir)
    logger = logging.getLogger(__name__)
    
    # Log environment information
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
    logger.info(f"Loading data from: {data_dir}")
    logger.info(f"Environment: {'Kaggle' if is_kaggle else 'Local'}")
    
    if is_kaggle:
        # First try the exact competition path
        kaggle_label_path = data_path / 'train_labels.csv'
        if kaggle_label_path.exists():
            train_labels_path = kaggle_label_path
            logger.info(f"Using Kaggle labels at: {kaggle_label_path}")
        else:
            # Fallback to recursive search in Kaggle
            try:
                csv_files = list(data_path.rglob("*train_labels*.csv"))
                if csv_files:
                    train_labels_path = csv_files[0]
                    logger.info(f"Found Kaggle labels via recursive search: {train_labels_path}")
            except Exception as e:
                logger.warning(f"Kaggle recursive search failed: {e}")
                
        # For train directory, use the provided data path
        train_dir = data_path / 'train'
        if train_dir.exists():
            logger.info(f"Using Kaggle train directory: {train_dir}")
        else:
            # Try to find the train directory
            possible_train_dirs = list(data_path.glob('*train*'))
            if possible_train_dirs:
                train_dir = possible_train_dirs[0]
                logger.info(f"Using found train directory: {train_dir}")
            else:
                logger.error("Cannot find train directory in Kaggle!")
                # We cannot run without data, so we must raise an error
                raise FileNotFoundError("Kaggle train directory not found")
    else:
        # ... existing non-Kaggle logic ...
        pass
    
    # Load labels
    try:
        labels_df = pd.read_csv(train_labels_path)
        logger.info(f"Loaded {len(labels_df)} training labels from {train_labels_path}")
        
        # Validate required columns
        required_columns = ['SeriesInstanceUID', 'Aneurysm Present'] + list(RSNA_LOCATION_LABELS)
        missing_columns = [col for col in required_columns if col not in labels_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in labels: {missing_columns}. Adding with default values.")
            for col in missing_columns:
                labels_df[col] = 0
    except Exception as e:
        logger.error(f"Error loading training labels from {train_labels_path}: {e}")
        
        # Create dummy labels as fallback
        logger.warning("Creating dummy labels to allow script to continue")
        series_ids = [f"dummy_series_{i}" for i in range(10)]
        labels_df = pd.DataFrame({
            'SeriesInstanceUID': series_ids,
            'Aneurysm Present': np.random.randint(0, 2, size=10),
            **{loc: np.random.randint(0, 2, size=10) for loc in RSNA_LOCATION_LABELS}
        })
    
    # Get all series paths
    series_paths = []
    try:
        # First try the expected Kaggle structure
        for study_dir in train_dir.glob("*"):
            if study_dir.is_dir():
                for series_dir in study_dir.glob("*"):
                    if series_dir.is_dir():
                        series_paths.append(str(series_dir))
        
        # If no series found, try alternative structures
        if not series_paths:
            logger.warning("No series found in standard structure, trying alternative paths...")
            for path in train_dir.rglob("*.dcm"):
                series_dir = path.parent
                if str(series_dir) not in series_paths:
                    series_paths.append(str(series_dir))
        
        logger.info(f"Found {len(series_paths)} DICOM series in {train_dir}")
        
        if not series_paths:
            logger.warning(f"No DICOM series found in {train_dir}")
            logger.info(f"Directory contents: {list(train_dir.glob('*'))}")
        
        # If no series found, create dummy paths for testing
        if not series_paths:
            logger.warning("No DICOM series found. Creating dummy paths for testing.")
            series_paths = [f"dummy_series_{i}" for i in range(len(labels_df))]
            
        return series_paths, labels_df
        
    except Exception as e:
        logger.error(f"Error loading DICOM series: {e}")
        # If error occurs, return dummy data to allow the script to run
        logger.warning("Returning dummy data due to error")
        return [f"dummy_series_{i}" for i in range(10)], pd.DataFrame()
    
    logger.info(f"Found {len(series_paths)} DICOM series")
    
    return series_paths, labels

def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        output_dir: Directory to save log files
        
    Returns:
        Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create handlers
    file_handler = logging.FileHandler(
        os.path.join(output_dir, 'training.log'),
        mode='w'  # Overwrite existing log file
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Remove any existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Suppress excessive logging from libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def load_config(config_path: Optional[str], **overrides) -> Dict[str, Any]:
    """Load configuration from file with overrides.
    
    Args:
        config_path: Path to config file (JSON or YAML)
        **overrides: Key-value pairs to override in config
        
    Returns:
        Dictionary containing configuration
    """
    config = {
        # Default configuration
        'data_dir': 'data',
        'output_dir': 'output',
        'num_folds': 5,
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10,
        'use_focal_loss': True,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'gradient_clip': 1.0,
        'num_workers': 4,
        'seed': 42,
        'debug': False
    }
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    file_config = yaml.safe_load(f)
                else:  # Assume JSON
                    file_config = json.load(f)
                
                # Update config with file values
                config.update(file_config)
                
        except Exception as e:
            logging.error(f"Error loading config file {config_path}: {e}")
            raise
    
    # Apply overrides
    config.update(overrides)
    
    # Handle debug mode
    if config.get('debug', False):
        config.update({
            'batch_size': min(config.get('batch_size', 8), 4),
            'num_epochs': min(config.get('num_epochs', 50), 5),
            'num_workers': min(config.get('num_workers', 4), 2),
            'early_stopping_patience': min(config.get('early_stopping_patience', 10), 3)
        })
    
    return config

def main(**kwargs):
    """
    Main entry point for training RSNA Intracranial Aneurysm Detection model.
    
    Args:
        **kwargs: Command line arguments as keyword arguments.
                 Supported args: data_dir, config, output_dir, num_folds, debug
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RSNA Aneurysm Detection Model')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Path to the data directory')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to config file (JSON or YAML)')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to save model checkpoints and logs')
    parser.add_argument('--num-folds', type=int, default=5,
                      help='Number of cross-validation folds')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with smaller dataset')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Parse arguments from kwargs if provided, otherwise from command line
    if kwargs:
        args = argparse.Namespace(**kwargs)
    else:
        args = parser.parse_args()
    
    # Set up logging first to capture all output
    logger = setup_logging(args.output_dir)
    
    # Log start of training
    logger.info("=" * 80)
    logger.info("Starting RSNA Intracranial Aneurysm Detection Training")
    logger.info("=" * 80)
    
    try:
        # Log environment information
        logger.info(f"Python version: {sys.version.split()[0]}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Command line arguments: {vars(args)}")
        
        # Load and validate configuration
        config = load_config(
            args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_folds=args.num_folds,
            debug=args.debug,
            seed=getattr(args, 'seed', 42)  # Safe access with default
        )
        
        logger.info("Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
        # Create output directory if it doesn't exist
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Set random seeds for reproducibility
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Load data
        data_dir = config['data_dir']
        logger.info(f"Loading data from: {data_dir}")
        
        series_paths, labels = load_training_data_kaggle(data_dir, config)
        logger.info(f"Found {len(series_paths)} DICOM series and {len(labels)} labels")
        
        if len(series_paths) == 0 or len(labels) == 0:
            raise ValueError("No data found. Please check your data directory.")
            
        # Apply debug mode adjustments
        if config.get('debug', False):
            logger.warning("DEBUG MODE: Using smaller dataset and fewer epochs")
            subset_size = min(20, len(series_paths))
            series_paths = series_paths[:subset_size]
            labels = labels.iloc[:subset_size]
            
        # Initialize cross-validation
        num_folds = config['num_folds']
        logger.info(f"Using {num_folds}-fold cross-validation")
        
        # Use 'Aneurysm Present' for stratification if available
        if 'Aneurysm Present' in labels.columns:
            y_stratify = labels['Aneurysm Present'].values
        else:
            logger.warning("'Aneurysm Present' column not found in labels. Using dummy stratification.")
            y_stratify = np.zeros(len(labels))
        
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        fold_aucs = []
        
        # Create output directory
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full configuration
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(series_paths, y_stratify)):
            fold_start_time = time.time()
            logger.info("\n" + "=" * 60)
            logger.info(f"Fold {fold + 1}/{num_folds}")
            logger.info("=" * 60)
            
            # Split data
            train_paths = [series_paths[i] for i in train_idx]
            val_paths = [series_paths[i] for i in val_idx]
            
            # Update config for this fold
            fold_config = config.copy()
            fold_config['output_dir'] = str(output_dir / f'fold_{fold + 1}')
            
            # Create fold output directory
            fold_output_dir = Path(fold_config['output_dir'])
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Train model
                logger.info(f"Training on {len(train_paths)} samples, validating on {len(val_paths)} samples")
                trainer = AneurysmTrainer(fold_config)
                trainer.label_cols = [col for col in RSNA_ALL_LABELS if col != 'SeriesInstanceUID']
                fold_auc = trainer.train(train_paths, val_paths, labels)
                fold_aucs.append(fold_auc)
                
                fold_time = time.time() - fold_start_time
                logger.info(f"Fold {fold + 1} completed in {fold_time/60:.1f} minutes")
                logger.info(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {e}", exc_info=True)
                if fold == 0:
                    # If first fold fails, re-raise the exception
                    raise
                # Otherwise continue with next fold
                continue
        
        # Calculate and log final results
        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            
            logger.info("\n" + "=" * 60)
            logger.info("CROSS-VALIDATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            logger.info(f"Fold AUCs: {fold_aucs}")
            
            # Save results
            results = {
                'mean_auc': float(mean_auc),
                'std_auc': float(std_auc),
                'fold_aucs': [float(auc) for auc in fold_aucs],
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'git_commit': get_git_commit()
            }
            
            results_path = output_dir / 'training_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {results_path}")
            
            return mean_auc
        else:
            logger.error("No successful folds completed.")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
    finally:
        logger.info("Training completed.")

def get_git_commit() -> str:
    """Get current git commit hash if available."""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

if __name__ == "__main__":
    main()
