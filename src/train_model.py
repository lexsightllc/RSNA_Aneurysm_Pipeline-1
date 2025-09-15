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
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
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
        self.model = Compact3DModel(num_classes=len(RSNA_ALL_LABELS) - 1)
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
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
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
    
    def train(self, train_paths: List[str], val_paths: List[str], labels: pd.DataFrame):
        """Main training loop."""
        train_loader, val_loader = self.create_data_loaders(train_paths, val_paths, labels)
        
        best_auc = 0.0
        patience_counter = 0
        max_patience = self.config.get('early_stopping_patience', 10)
        
        for epoch in range(self.config.get('num_epochs', 100)):
            logger.info(f"Epoch {epoch + 1}/{self.config.get('num_epochs', 100)}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_auc = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_auc)
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                
                # Save model
                save_path = Path(self.config.get('output_dir', 'models')) / f'best_model_fold.pth'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'config': self.config
                }, save_path)
                
                logger.info(f"New best model saved with AUC: {best_auc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        return best_auc

def load_training_data(data_dir: str):
    """Load training data paths and labels.
    
    Args:
        data_dir: Path to the directory containing training data
        
    Returns:
        tuple: (list_of_series_paths, labels_dataframe)
    """
    data_path = Path(data_dir)
    
    # Check if running in Kaggle environment
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    if is_kaggle:
        # Kaggle specific paths
        train_labels_path = data_path / "train_labels.csv"
        train_dir = data_path / "train"
    else:
        # Local development paths
        train_labels_path = data_path / "train" / "train_labels.csv"
        train_dir = data_path / "train"
    
    # Load labels
    try:
        labels_df = pd.read_csv(train_labels_path)
        logger.info(f"Loaded {len(labels_df)} training labels from {train_labels_path}")
    except Exception as e:
        logger.error(f"Error loading training labels: {e}")
        raise
    
    # Get all series paths
    series_paths = []
    try:
        if is_kaggle:
            # In Kaggle, the structure is different
            for study_dir in train_dir.glob("*"):
                if study_dir.is_dir():
                    series_paths.extend([str(p) for p in study_dir.glob("*") if p.is_dir()])
        else:
            # Local structure
            for study_dir in train_dir.glob("*"):
                if study_dir.is_dir():
                    series_paths.extend([str(p) for p in study_dir.glob("*") if p.is_dir()])
        
        logger.info(f"Found {len(series_paths)} DICOM series")
        
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

def main(**kwargs):
    """
    Main entry point for training.
    
    Args:
        **kwargs: Can include any of the command line arguments as keyword arguments.
                 Supported args: data_dir, config, output_dir, num_folds, debug
    """
    import argparse
    
    # Check if running in Kaggle environment
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    # Default config path
    default_config_path = '/kaggle/input/rsna-aneurysm-pipeline-1/config/pipeline_config.json' if is_kaggle else 'config/pipeline_config.json'
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train RSNA Aneurysm Detection Model')
    parser.add_argument('--data-dir', type=str, required=not is_kaggle,
                      help='Path to the directory containing training data')
    parser.add_argument('--config', type=str, default=default_config_path,
                      help=f'Path to config JSON file (default: {default_config_path})')
    parser.add_argument('--output-dir', type=str, 
                      default='/kaggle/working' if is_kaggle else './output',
                      help='Directory to save model checkpoints and logs')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    
    # If kwargs are provided, use those, otherwise parse from command line
    if kwargs:
        # Ensure debug flag is properly set if passed as a boolean
        if 'debug' in kwargs and isinstance(kwargs['debug'], bool):
            if kwargs['debug']:
                kwargs['debug'] = True
            else:
                del kwargs['debug']  # Let it use the default (False)
        args = parser.parse_args(args=[])
        for key, value in kwargs.items():
            setattr(args, key.replace('-', '_'), value)
    else:
        args = parser.parse_args()
    
    # Default configuration
    config = {
        'batch_size': 2 if getattr(args, 'debug', False) else 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 5 if getattr(args, 'debug', False) else 100,
        'input_size': [128, 128, 128],
        'use_focal_loss': True,
        'focal_alpha': 1,
        'focal_gamma': 2,
        'use_weighted_sampling': True,
        'gradient_clip': 1.0,
        'early_stopping_patience': 10,
        'num_workers': 0 if getattr(args, 'debug', False) else 4,
        'output_dir': args.output_dir
    }
    
    # Load custom config if provided
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        config.update(custom_config)
    
    # Load data
    series_paths, labels = load_training_data(args.data_dir)
    
    if getattr(args, 'debug', False):
        # Use smaller subset for debugging
        logger.info("Debug mode: Using smaller dataset (20 samples)")
        series_paths = series_paths[:20]
        labels = labels.head(20)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    
    # Use 'Aneurysm Present' for stratification
    y_stratify = labels['Aneurysm Present'].values
    
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(series_paths, y_stratify)):
        logger.info(f"\n=== Fold {fold + 1}/{args.num_folds} ===")
        
        # Split data
        train_paths = [series_paths[i] for i in train_idx]
        val_paths = [series_paths[i] for i in val_idx]
        
        # Update config for this fold
        fold_config = config.copy()
        fold_config['output_dir'] = f"{args.output_dir}/fold_{fold + 1}"
        
        # Train model
        trainer = AneurysmTrainer(fold_config)
        fold_auc = trainer.train(train_paths, val_paths, labels)
        fold_aucs.append(fold_auc)
        
        logger.info(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
    
    # Final results
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    logger.info(f"\n=== Final Results ===")
    logger.info(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    logger.info(f"Fold AUCs: {fold_aucs}")
    
    # Save results
    results = {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'fold_aucs': fold_aucs,
        'config': config
    }
    
    results_path = Path(args.output_dir) / 'training_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
