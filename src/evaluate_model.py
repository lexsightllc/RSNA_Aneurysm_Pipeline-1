#!/usr/bin/env python3
"""
RSNA Intracranial Aneurysm Detection - Evaluation Framework

Comprehensive evaluation with:
- Multi-class metrics (AUC, AP, F1)
- Location-specific analysis
- Calibration assessment
- Error analysis
- Knowledge-based validation
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, calibration_curve
)
from sklearn.calibration import CalibratedClassifierCV
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from src.utils.prediction_utils import RSNA_ALL_LABELS, RSNA_LOCATION_LABELS
from src.adapter_compact3d_rc2 import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AneurysmEvaluator:
    """Comprehensive evaluation framework for aneurysm detection."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Location-specific knowledge from domain data
        self.location_prevalence = {
            'Anterior Communicating Artery': 0.325,  # 30-35%
            'Left Posterior Communicating Artery': 0.325,
            'Right Posterior Communicating Artery': 0.325,
            'Left Middle Cerebral Artery': 0.10,
            'Right Middle Cerebral Artery': 0.10,
            'Basilar Tip': 0.05,
            'Other Posterior Circulation': 0.02,
            'Left Infraclinoid Internal Carotid Artery': 0.05,
            'Right Infraclinoid Internal Carotid Artery': 0.05,
            'Left Supraclinoid Internal Carotid Artery': 0.08,
            'Right Supraclinoid Internal Carotid Artery': 0.08,
            'Left Anterior Cerebral Artery': 0.03,
            'Right Anterior Cerebral Artery': 0.03,
        }
        
        # Clinical significance weights
        self.clinical_weights = {
            'Anterior Communicating Artery': 1.0,  # High rupture risk
            'Left Posterior Communicating Artery': 1.0,
            'Right Posterior Communicating Artery': 1.0,
            'Left Middle Cerebral Artery': 0.9,
            'Right Middle Cerebral Artery': 0.9,
            'Basilar Tip': 1.1,  # Higher risk of giant aneurysms
            'Other Posterior Circulation': 0.8,
            'Left Infraclinoid Internal Carotid Artery': 0.7,
            'Right Infraclinoid Internal Carotid Artery': 0.7,
            'Left Supraclinoid Internal Carotid Artery': 0.8,
            'Right Supraclinoid Internal Carotid Artery': 0.8,
            'Left Anterior Cerebral Artery': 0.6,
            'Right Anterior Cerebral Artery': 0.6,
        }
    
    def evaluate_predictions(self, 
                           predictions: pd.DataFrame, 
                           ground_truth: pd.DataFrame,
                           save_plots: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model predictions.
        
        Args:
            predictions: DataFrame with model predictions
            ground_truth: DataFrame with true labels
            save_plots: Whether to save evaluation plots
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}
        
        # Merge predictions and ground truth
        merged = pd.merge(
            predictions, ground_truth, 
            on='SeriesInstanceUID', 
            suffixes=('_pred', '_true')
        )
        
        if len(merged) == 0:
            logger.error("No matching series found between predictions and ground truth")
            return {}
        
        logger.info(f"Evaluating {len(merged)} cases")
        
        # Overall metrics
        results['overall'] = self._evaluate_overall_performance(merged)
        
        # Location-specific metrics
        results['location_specific'] = self._evaluate_location_performance(merged)
        
        # Calibration analysis
        results['calibration'] = self._evaluate_calibration(merged)
        
        # Error analysis
        results['error_analysis'] = self._analyze_errors(merged)
        
        # Clinical relevance analysis
        results['clinical_analysis'] = self._analyze_clinical_relevance(merged)
        
        # Generate plots
        if save_plots:
            self._generate_evaluation_plots(merged, results)
        
        # Save detailed results
        self._save_results(results)
        
        return results
    
    def _evaluate_overall_performance(self, merged: pd.DataFrame) -> Dict[str, float]:
        """Evaluate overall model performance."""
        metrics = {}
        
        # Aneurysm Present metrics
        y_true = merged['Aneurysm Present_true'].values
        y_pred = merged['Aneurysm Present_pred'].values
        
        try:
            metrics['aneurysm_present_auc'] = roc_auc_score(y_true, y_pred)
            metrics['aneurysm_present_ap'] = average_precision_score(y_true, y_pred)
        except ValueError as e:
            logger.warning(f"Could not calculate AUC/AP: {e}")
            metrics['aneurysm_present_auc'] = 0.5
            metrics['aneurysm_present_ap'] = sum(y_true) / len(y_true)
        
        # Binary classification metrics at optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics['optimal_threshold'] = optimal_threshold
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)
        metrics['specificity'] = np.sum((y_true == 0) & (y_pred_binary == 0)) / np.sum(y_true == 0)
        
        # Prevalence and balance
        metrics['prevalence'] = np.mean(y_true)
        metrics['predicted_prevalence'] = np.mean(y_pred_binary)
        
        logger.info(f"Overall AUC: {metrics['aneurysm_present_auc']:.3f}")
        logger.info(f"Overall AP: {metrics['aneurysm_present_ap']:.3f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.3f}")
        
        return metrics
    
    def _evaluate_location_performance(self, merged: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate performance for each anatomical location."""
        location_metrics = {}
        
        for location in RSNA_LOCATION_LABELS:
            if f"{location}_true" not in merged.columns or f"{location}_pred" not in merged.columns:
                continue
                
            y_true = merged[f"{location}_true"].values
            y_pred = merged[f"{location}_pred"].values
            
            if len(np.unique(y_true)) < 2:
                # Skip if all labels are the same
                continue
            
            try:
                auc = roc_auc_score(y_true, y_pred)
                ap = average_precision_score(y_true, y_pred)
                
                location_metrics[location] = {
                    'auc': auc,
                    'ap': ap,
                    'prevalence': np.mean(y_true),
                    'predicted_prevalence': np.mean(y_pred > 0.5),
                    'expected_prevalence': self.location_prevalence.get(location, 0.05),
                    'clinical_weight': self.clinical_weights.get(location, 1.0)
                }
                
            except ValueError as e:
                logger.warning(f"Could not calculate metrics for {location}: {e}")
        
        # Sort by clinical importance
        sorted_locations = sorted(
            location_metrics.items(),
            key=lambda x: x[1]['clinical_weight'] * x[1]['auc'],
            reverse=True
        )
        
        logger.info("Location-specific performance (top 5):")
        for location, metrics in sorted_locations[:5]:
            logger.info(f"  {location}: AUC={metrics['auc']:.3f}, AP={metrics['ap']:.3f}")
        
        return dict(sorted_locations)
    
    def _evaluate_calibration(self, merged: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model calibration."""
        calibration_results = {}
        
        y_true = merged['Aneurysm Present_true'].values
        y_pred = merged['Aneurysm Present_pred'].values
        
        # Calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred, n_bins=10
            )
            
            # Brier score
            brier_score = np.mean((y_pred - y_true) ** 2)
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            calibration_results = {
                'brier_score': brier_score,
                'ece': ece,
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist(),
                'is_well_calibrated': ece < 0.1  # ECE < 0.1 is considered well-calibrated
            }
            
            logger.info(f"Calibration - Brier Score: {brier_score:.3f}, ECE: {ece:.3f}")
            
        except Exception as e:
            logger.warning(f"Could not calculate calibration metrics: {e}")
            calibration_results = {'error': str(e)}
        
        return calibration_results
    
    def _analyze_errors(self, merged: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction errors."""
        y_true = merged['Aneurysm Present_true'].values
        y_pred = merged['Aneurysm Present_pred'].values
        
        # Use optimal threshold from overall evaluation
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        
        # Error categories
        true_positives = (y_true == 1) & (y_pred_binary == 1)
        false_positives = (y_true == 0) & (y_pred_binary == 1)
        true_negatives = (y_true == 0) & (y_pred_binary == 0)
        false_negatives = (y_true == 1) & (y_pred_binary == 0)
        
        error_analysis = {
            'true_positives': int(np.sum(true_positives)),
            'false_positives': int(np.sum(false_positives)),
            'true_negatives': int(np.sum(true_negatives)),
            'false_negatives': int(np.sum(false_negatives)),
            'total_cases': len(merged)
        }
        
        # Confidence analysis for errors
        if np.sum(false_positives) > 0:
            fp_confidences = y_pred[false_positives]
            error_analysis['fp_mean_confidence'] = float(np.mean(fp_confidences))
            error_analysis['fp_std_confidence'] = float(np.std(fp_confidences))
        
        if np.sum(false_negatives) > 0:
            fn_confidences = y_pred[false_negatives]
            error_analysis['fn_mean_confidence'] = float(np.mean(fn_confidences))
            error_analysis['fn_std_confidence'] = float(np.std(fn_confidences))
        
        logger.info(f"Error Analysis - FP: {error_analysis['false_positives']}, "
                   f"FN: {error_analysis['false_negatives']}")
        
        return error_analysis
    
    def _analyze_clinical_relevance(self, merged: pd.DataFrame) -> Dict[str, Any]:
        """Analyze clinical relevance of predictions."""
        clinical_analysis = {}
        
        # High-risk locations (ACom, PCom, MCA bifurcation, Basilar tip)
        high_risk_locations = [
            'Anterior Communicating Artery',
            'Left Posterior Communicating Artery',
            'Right Posterior Communicating Artery',
            'Left Middle Cerebral Artery',
            'Right Middle Cerebral Artery',
            'Basilar Tip'
        ]
        
        # Calculate performance on high-risk locations
        high_risk_true = []
        high_risk_pred = []
        
        for location in high_risk_locations:
            if f"{location}_true" in merged.columns:
                high_risk_true.extend(merged[f"{location}_true"].values)
                high_risk_pred.extend(merged[f"{location}_pred"].values)
        
        if len(high_risk_true) > 0:
            try:
                high_risk_auc = roc_auc_score(high_risk_true, high_risk_pred)
                clinical_analysis['high_risk_locations_auc'] = high_risk_auc
            except ValueError:
                clinical_analysis['high_risk_locations_auc'] = 0.5
        
        # Prevalence consistency check
        prevalence_consistency = {}
        for location in RSNA_LOCATION_LABELS:
            if f"{location}_true" in merged.columns:
                observed_prev = merged[f"{location}_true"].mean()
                expected_prev = self.location_prevalence.get(location, 0.05)
                predicted_prev = (merged[f"{location}_pred"] > 0.5).mean()
                
                prevalence_consistency[location] = {
                    'observed': observed_prev,
                    'expected': expected_prev,
                    'predicted': predicted_prev,
                    'obs_vs_exp_ratio': observed_prev / (expected_prev + 1e-8),
                    'pred_vs_obs_ratio': predicted_prev / (observed_prev + 1e-8)
                }
        
        clinical_analysis['prevalence_consistency'] = prevalence_consistency
        
        return clinical_analysis
    
    def _generate_evaluation_plots(self, merged: pd.DataFrame, results: Dict[str, Any]):
        """Generate evaluation plots."""
        # ROC Curve
        plt.figure(figsize=(15, 10))
        
        # Overall ROC
        plt.subplot(2, 3, 1)
        y_true = merged['Aneurysm Present_true'].values
        y_pred = merged['Aneurysm Present_pred'].values
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = results['overall']['aneurysm_present_auc']
        
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Aneurysm Present')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        plt.subplot(2, 3, 2)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = results['overall']['aneurysm_present_ap']
        
        plt.plot(recall, precision, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calibration Plot
        plt.subplot(2, 3, 3)
        if 'fraction_of_positives' in results['calibration']:
            fop = results['calibration']['fraction_of_positives']
            mpv = results['calibration']['mean_predicted_value']
            
            plt.plot(mpv, fop, 's-', label='Model')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Location Performance
        plt.subplot(2, 3, 4)
        if 'location_specific' in results:
            locations = list(results['location_specific'].keys())[:10]  # Top 10
            aucs = [results['location_specific'][loc]['auc'] for loc in locations]
            
            plt.barh(range(len(locations)), aucs)
            plt.yticks(range(len(locations)), [loc.replace(' ', '\n') for loc in locations])
            plt.xlabel('AUC')
            plt.title('Location-Specific Performance')
            plt.grid(True, alpha=0.3)
        
        # Prediction Distribution
        plt.subplot(2, 3, 5)
        pos_preds = y_pred[y_true == 1]
        neg_preds = y_pred[y_true == 0]
        
        plt.hist(neg_preds, bins=20, alpha=0.7, label='Negative', density=True)
        plt.hist(pos_preds, bins=20, alpha=0.7, label='Positive', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion Matrix
        plt.subplot(2, 3, 6)
        threshold = results['overall']['optimal_threshold']
        y_pred_binary = (y_pred >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix (threshold={threshold:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {self.output_dir / 'evaluation_plots.png'}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON."""
        results_path = self.output_dir / 'evaluation_results.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_numpy(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RSNA Aneurysm Detection Model')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV file')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth CSV file')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load data
    predictions = pd.read_csv(args.predictions)
    ground_truth = pd.read_csv(args.ground_truth)
    
    logger.info(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth labels")
    
    # Evaluate
    evaluator = AneurysmEvaluator(args.output_dir)
    results = evaluator.evaluate_predictions(predictions, ground_truth)
    
    # Print summary
    if 'overall' in results:
        print(f"\n=== Evaluation Summary ===")
        print(f"Overall AUC: {results['overall']['aneurysm_present_auc']:.4f}")
        print(f"Overall AP: {results['overall']['aneurysm_present_ap']:.4f}")
        print(f"F1 Score: {results['overall']['f1_score']:.4f}")
        print(f"Precision: {results['overall']['precision']:.4f}")
        print(f"Recall: {results['overall']['recall']:.4f}")
        
        if 'calibration' in results and 'ece' in results['calibration']:
            print(f"Expected Calibration Error: {results['calibration']['ece']:.4f}")

if __name__ == "__main__":
    main()
