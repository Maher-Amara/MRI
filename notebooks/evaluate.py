"""
Evaluation Script for Brain Tumor Detection using EfficientNet
Phase 5 of CRISP-DM Methodology

This script:
1. Loads a trained model
2. Evaluates on test set
3. Calculates comprehensive metrics (accuracy, precision, recall, F1-score)
4. Generates confusion matrix
5. Creates visualizations (confusion matrix, ROC curves, PR curves)
6. Saves results and images to docs/assets

Usage:
    python evaluate.py --model-path models/efficientnet/v1.0/efficientnet_b2/best_model.h5
    python evaluate.py --model-path models/efficientnet/v1.0/efficientnet_b2/best_model.h5 --model-variant b2
"""

import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    plt = None
    sns = None
    print("Warning: Matplotlib/Seaborn not available. Visualizations will not be generated.")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# Import data preparation
from datapreparation import DataPreparation


class ModelEvaluator:
    """Evaluator class for trained EfficientNet models."""
    
    def __init__(self, model_path, model_variant='b2', dataset_path='../dataset'):
        """
        Initialize the model evaluator.
        
        Args:
            model_path (str): Path to trained model file (.keras or .h5)
            model_variant (str): EfficientNet variant ('b2', 'b3', or 'b4')
            dataset_path (str): Path to dataset directory
        """
        # Resolve model path relative to notebooks directory
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            # If relative, try resolving from notebooks directory first
            notebooks_dir = Path(__file__).parent
            resolved_path = (notebooks_dir / model_path).resolve()
            if not resolved_path.exists():
                # Try with ../ prefix (relative to notebooks)
                resolved_path = (notebooks_dir.parent / model_path).resolve()
            self.model_path = resolved_path
        else:
            self.model_path = model_path_obj
        
        self.model_variant = model_variant.lower()
        self.dataset_path = Path(dataset_path)
        
        # Load model - try .keras first, then .h5
        print(f"\n{'='*80}")
        print(f"LOADING MODEL")
        print(f"{'='*80}")
        print(f"Model path: {self.model_path}")
        
        # Determine which file exists
        model_file = None
        if self.model_path.suffix in ['.keras', '.h5']:
            # User specified exact file
            if self.model_path.exists():
                model_file = self.model_path
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        else:
            # Try both formats (prefer .keras)
            keras_path = self.model_path.with_suffix('.keras')
            h5_path = self.model_path.with_suffix('.h5')
            
            if keras_path.exists():
                model_file = keras_path
            elif h5_path.exists():
                model_file = h5_path
            else:
                # Try as directory and look for best_model.keras or best_model.h5
                if self.model_path.is_dir():
                    keras_path = self.model_path / 'best_model.keras'
                    h5_path = self.model_path / 'best_model.h5'
                    if keras_path.exists():
                        model_file = keras_path
                    elif h5_path.exists():
                        model_file = h5_path
                
                if model_file is None:
                    raise FileNotFoundError(
                        f"Model file not found. Tried:\n"
                        f"  - {keras_path}\n"
                        f"  - {h5_path}\n"
                        f"  - {self.model_path}/best_model.keras\n"
                        f"  - {self.model_path}/best_model.h5"
                    )
        
        # Load the model
        self.model = keras.models.load_model(str(model_file))
        self.model_path = model_file  # Update to actual file used
        
        print(f"✓ Model loaded successfully")
        
        # Get model configuration
        model_dir = self.model_path.parent
        config_path = model_dir / 'model_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.model_variant = self.config.get('model_variant', model_variant)
            self.version = self.config.get('version', 'v1.0')
            self.class_names = self.config.get('class_names', [])
            self.class_indices = self.config.get('class_indices', {})
        else:
            self.config = {}
            self.version = 'v1.0'
            self.class_names = []
            self.class_indices = {}
        
        # Initialize data preparation
        self.data_prep = DataPreparation(
            dataset_path=str(self.dataset_path),
            model_variant=self.model_variant,
            create_splits=False
        )
        
        # Create test data generator
        _, _, self.test_gen = self.data_prep.create_data_generators(
            batch_size=32,
            seed=42,  # Fixed seed for consistent evaluation
            validation_split=0.2,
            use_merged=True
        )
        
        # Get class names from generator if not in config
        if not self.class_names:
            self.class_names = list(self.test_gen.class_indices.keys())
            self.class_indices = self.test_gen.class_indices
        
        print(f"\nClasses: {self.class_names}")
        print(f"Test samples: {self.test_gen.samples}")
        
        # Setup output directory
        self.assets_dir = Path('../docs/assets')
        self.assets_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
    
    def evaluate(self):
        """Evaluate the model and calculate all metrics."""
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL")
        print(f"{'='*80}")
        
        # Get predictions
        print("Generating predictions...")
        self.y_pred_proba = self.model.predict(self.test_gen, verbose=1)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        
        # Get true labels
        self.y_true = self.test_gen.classes[:len(self.y_pred)]
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Print results
        self._print_results()
        
        return self.metrics
    
    def _calculate_metrics(self):
        """Calculate all evaluation metrics."""
        # Overall accuracy
        accuracy = accuracy_score(self.y_true, self.y_pred)
        
        # Per-class and averaged metrics
        precision_macro = precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        precision_per_class = precision_score(self.y_true, self.y_pred, average=None, zero_division=0)
        
        recall_macro = recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        recall_per_class = recall_score(self.y_true, self.y_pred, average=None, zero_division=0)
        
        f1_macro = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        f1_per_class = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # ROC AUC (one-vs-rest for multi-class)
        y_true_binarized = label_binarize(self.y_true, classes=range(len(self.class_names)))
        if len(self.class_names) == 2:
            roc_auc = roc_auc_score(self.y_true, self.y_pred_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_true_binarized, self.y_pred_proba, average='macro', multi_class='ovr')
        
        # Average Precision
        if len(self.class_names) == 2:
            avg_precision = average_precision_score(self.y_true, self.y_pred_proba[:, 1])
        else:
            avg_precision = average_precision_score(y_true_binarized, self.y_pred_proba, average='macro')
        
        # Store metrics
        self.metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'precision_weighted': float(precision_weighted),
            'precision_per_class': precision_per_class.tolist(),
            'recall_macro': float(recall_macro),
            'recall_weighted': float(recall_weighted),
            'recall_per_class': recall_per_class.tolist(),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_per_class': f1_per_class.tolist(),
            'roc_auc': float(roc_auc),
            'average_precision': float(avg_precision),
            'confusion_matrix': cm.tolist()
        }
        
        self.cm = cm
    
    def _print_results(self):
        """Print evaluation results."""
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*80}")
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:        {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)")
        print(f"  Precision (Macro): {self.metrics['precision_macro']:.4f}")
        print(f"  Precision (Weighted): {self.metrics['precision_weighted']:.4f}")
        print(f"  Recall (Macro):   {self.metrics['recall_macro']:.4f}")
        print(f"  Recall (Weighted): {self.metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (Macro): {self.metrics['f1_macro']:.4f}")
        print(f"  F1-Score (Weighted): {self.metrics['f1_weighted']:.4f}")
        print(f"  ROC-AUC:          {self.metrics['roc_auc']:.4f}")
        print(f"  Average Precision: {self.metrics['average_precision']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 60)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<20} {self.metrics['precision_per_class'][i]:<12.4f} "
                  f"{self.metrics['recall_per_class'][i]:<12.4f} {self.metrics['f1_per_class'][i]:<12.4f}")
        
        print(f"\nConfusion Matrix:")
        print(self.cm)
        
        # Classification report
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION REPORT")
        print(f"{'='*80}")
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names,
            digits=4
        )
        print(report)
    
    def visualize_confusion_matrix(self):
        """Create and save confusion matrix visualization."""
        if plt is None:
            print("⚠ Matplotlib not available. Skipping confusion matrix visualization.")
            return
        
        print(f"\n{'='*80}")
        print(f"VISUALIZING CONFUSION MATRIX")
        print(f"{'='*80}")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('Actual', fontsize=12)
        
        # Normalized (percentages)
        cm_normalized = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        cm_path = self.assets_dir / f'confusion_matrix_{self.model_variant}_{self.version}.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {cm_path}")
        plt.close()
    
    def visualize_roc_curves(self):
        """Create and save ROC curve visualization."""
        if plt is None:
            print("⚠ Matplotlib not available. Skipping ROC curve visualization.")
            return
        
        print(f"\n{'='*80}")
        print(f"VISUALIZING ROC CURVES")
        print(f"{'='*80}")
        
        # Binarize labels for multi-class
        y_true_binarized = label_binarize(self.y_true, classes=range(len(self.class_names)))
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if len(self.class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        else:
            # Multi-class: one-vs-rest
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            for i, class_name in enumerate(self.class_names):
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], self.y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                       label=f'{class_name} (AUC = {roc_auc:.4f})')
        
        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves - EfficientNet-{self.model_variant.upper()} (v{self.version})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        roc_path = self.assets_dir / f'roc_curves_{self.model_variant}_{self.version}.png'
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        print(f"✓ ROC curves saved to: {roc_path}")
        plt.close()
    
    def visualize_pr_curves(self):
        """Create and save Precision-Recall curve visualization."""
        if plt is None:
            print("⚠ Matplotlib not available. Skipping PR curve visualization.")
            return
        
        print(f"\n{'='*80}")
        print(f"VISUALIZING PRECISION-RECALL CURVES")
        print(f"{'='*80}")
        
        # Binarize labels for multi-class
        y_true_binarized = label_binarize(self.y_true, classes=range(len(self.class_names)))
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if len(self.class_names) == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba[:, 1])
            avg_precision = average_precision_score(self.y_true, self.y_pred_proba[:, 1])
            ax.plot(recall, precision, lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
        else:
            # Multi-class: one-vs-rest
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            for i, class_name in enumerate(self.class_names):
                precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], self.y_pred_proba[:, i])
                avg_precision = average_precision_score(y_true_binarized[:, i], self.y_pred_proba[:, i])
                ax.plot(recall, precision, lw=2, color=colors[i % len(colors)],
                       label=f'{class_name} (AP = {avg_precision:.4f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curves - EfficientNet-{self.model_variant.upper()} (v{self.version})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        pr_path = self.assets_dir / f'pr_curves_{self.model_variant}_{self.version}.png'
        plt.savefig(pr_path, dpi=150, bbox_inches='tight')
        print(f"✓ Precision-Recall curves saved to: {pr_path}")
        plt.close()
    
    def visualize_metrics_summary(self):
        """Create and save metrics summary visualization."""
        if plt is None:
            print("⚠ Matplotlib not available. Skipping metrics summary visualization.")
            return
        
        print(f"\n{'='*80}")
        print(f"VISUALIZING METRICS SUMMARY")
        print(f"{'='*80}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Evaluation Metrics Summary - EfficientNet-{self.model_variant.upper()} (v{self.version})',
                    fontsize=16, fontweight='bold')
        
        # Overall metrics bar chart
        overall_metrics = {
            'Accuracy': self.metrics['accuracy'],
            'Precision\n(Macro)': self.metrics['precision_macro'],
            'Recall\n(Macro)': self.metrics['recall_macro'],
            'F1-Score\n(Macro)': self.metrics['f1_macro'],
            'ROC-AUC': self.metrics['roc_auc'],
            'Avg Precision': self.metrics['average_precision']
        }
        
        axes[0, 0].bar(range(len(overall_metrics)), list(overall_metrics.values()), 
                      color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'])
        axes[0, 0].set_xticks(range(len(overall_metrics)))
        axes[0, 0].set_xticklabels(list(overall_metrics.keys()), rotation=45, ha='right')
        axes[0, 0].set_ylabel('Score', fontsize=12)
        axes[0, 0].set_title('Overall Metrics', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, (key, val) in enumerate(overall_metrics.items()):
            axes[0, 0].text(i, val, f'{val:.3f}', ha='center', va='bottom')
        
        # Per-class precision
        axes[0, 1].bar(range(len(self.class_names)), self.metrics['precision_per_class'],
                       color='#3498db')
        axes[0, 1].set_xticks(range(len(self.class_names)))
        axes[0, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Precision', fontsize=12)
        axes[0, 1].set_title('Per-Class Precision', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis='y', alpha=0.3)
        for i, val in enumerate(self.metrics['precision_per_class']):
            axes[0, 1].text(i, val, f'{val:.3f}', ha='center', va='bottom')
        
        # Per-class recall
        axes[1, 0].bar(range(len(self.class_names)), self.metrics['recall_per_class'],
                      color='#e74c3c')
        axes[1, 0].set_xticks(range(len(self.class_names)))
        axes[1, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Recall', fontsize=12)
        axes[1, 0].set_title('Per-Class Recall', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, val in enumerate(self.metrics['recall_per_class']):
            axes[1, 0].text(i, val, f'{val:.3f}', ha='center', va='bottom')
        
        # Per-class F1-score
        axes[1, 1].bar(range(len(self.class_names)), self.metrics['f1_per_class'],
                      color='#2ecc71')
        axes[1, 1].set_xticks(range(len(self.class_names)))
        axes[1, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('F1-Score', fontsize=12)
        axes[1, 1].set_title('Per-Class F1-Score', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, val in enumerate(self.metrics['f1_per_class']):
            axes[1, 1].text(i, val, f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        summary_path = self.assets_dir / f'metrics_summary_{self.model_variant}_{self.version}.png'
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"✓ Metrics summary saved to: {summary_path}")
        plt.close()
    
    def save_results(self):
        """Save evaluation results to JSON file."""
        results_path = self.model_path.parent / 'evaluation_results.json'
        
        results = {
            'model_path': str(self.model_path),
            'model_variant': self.model_variant,
            'version': self.version,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_samples': int(len(self.y_true)),
            'metrics': self.metrics,
            'class_names': self.class_names
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Evaluation results saved to: {results_path}")
        return results_path
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE MODEL EVALUATION")
        print(f"{'='*80}")
        print(f"Model: EfficientNet-{self.model_variant.upper()}")
        print(f"Version: {self.version}")
        print(f"Model Path: {self.model_path}")
        
        # Evaluate
        self.evaluate()
        
        # Generate visualizations
        self.visualize_confusion_matrix()
        self.visualize_roc_curves()
        self.visualize_pr_curves()
        self.visualize_metrics_summary()
        
        # Save results
        self.save_results()
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nGenerated Files:")
        print(f"  - Confusion Matrix: docs/assets/confusion_matrix_{self.model_variant}_{self.version}.png")
        print(f"  - ROC Curves: docs/assets/roc_curves_{self.model_variant}_{self.version}.png")
        print(f"  - PR Curves: docs/assets/pr_curves_{self.model_variant}_{self.version}.png")
        print(f"  - Metrics Summary: docs/assets/metrics_summary_{self.model_variant}_{self.version}.png")
        print(f"  - Results JSON: {self.model_path.parent}/evaluation_results.json")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate EfficientNet model for Brain Tumor Detection')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model file (.keras or .h5)')
    parser.add_argument('--model-variant', type=str, default='b2', choices=['b2', 'b3', 'b4'],
                       help='EfficientNet variant (default: b2, auto-detected from config if available)')
    parser.add_argument('--dataset', type=str, default='../dataset',
                       help='Dataset path (default: ../dataset)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"BRAIN TUMOR DETECTION - MODEL EVALUATION")
    print(f"{'='*80}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        model_variant=args.model_variant,
        dataset_path=args.dataset
    )
    
    # Run full evaluation
    evaluator.run_full_evaluation()
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
