"""
Training Script for Brain Tumor Detection using EfficientNet
Phase 4 of CRISP-DM Methodology

This script:
1. Loads preprocessed data using DataPreparation
2. Creates EfficientNet models with transfer learning
3. Trains models with proper callbacks and class weights
4. Saves models with versioning
5. Evaluates and saves training history

Usage:
    python main.py --model b2 --version v1.0 --epochs 50
"""

import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime
import warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    CSVLogger, TensorBoard
)

# Suppress HDF5 warnings (we use .keras format now)
warnings.filterwarnings('ignore', category=UserWarning, message='.*HDF5.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    plt = None
    print("Warning: Matplotlib not available. Training curves will not be visualized.")

# Import data preparation
from datapreparation import DataPreparation

# Set memory growth for GPU if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU available: {len(gpus)} device(s)")
except Exception as e:
    print(f"GPU configuration note: {e}")


class EfficientNetTrainer:
    """Trainer class for EfficientNet models on brain tumor detection."""
    
    # EfficientNet model variants and their input sizes
    EFFICIENTNET_VARIANTS = {
        'b2': {'size': 260, 'name': 'EfficientNetB2'},
        'b3': {'size': 300, 'name': 'EfficientNetB3'},
        'b4': {'size': 380, 'name': 'EfficientNetB4'}
    }
    
    def __init__(self, model_variant='b2', version='v1.0', dataset_path='../dataset',
                 base_model_trainable=False, dropout_rate=0.2):
        """
        Initialize the EfficientNet trainer.
        
        Args:
            model_variant (str): EfficientNet variant ('b2', 'b3', or 'b4')
            version (str): Model version for tracking (e.g., 'v1.0', 'v1.1')
            dataset_path (str): Path to dataset directory
            base_model_trainable (bool): Whether to fine-tune base EfficientNet layers
            dropout_rate (float): Dropout rate for classification head
        """
        if model_variant.lower() not in self.EFFICIENTNET_VARIANTS:
            raise ValueError(f"Model variant must be one of {list(self.EFFICIENTNET_VARIANTS.keys())}")
        
        self.model_variant = model_variant.lower()
        self.version = version
        self.dataset_path = Path(dataset_path)
        self.base_model_trainable = base_model_trainable
        self.dropout_rate = dropout_rate
        
        # Get model configuration
        self.input_size = self.EFFICIENTNET_VARIANTS[self.model_variant]['size']
        self.model_name = self.EFFICIENTNET_VARIANTS[self.model_variant]['name']
        
        # Setup paths
        self.models_dir = Path('../models/efficientnet') / version / f'efficientnet_{self.model_variant}'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data preparation
        print(f"\n{'='*80}")
        print(f"INITIALIZING TRAINER")
        print(f"{'='*80}")
        print(f"Model: EfficientNet-{self.model_variant.upper()}")
        print(f"Version: {version}")
        print(f"Input Size: {self.input_size}×{self.input_size}")
        print(f"Base Model Trainable: {base_model_trainable}")
        print(f"Models Directory: {self.models_dir}")
        
        self.data_prep = DataPreparation(
            dataset_path=str(self.dataset_path),
            model_variant=self.model_variant,
            create_splits=False  # Don't create fixed splits, use random during training
        )
        
        # Create data generators with random splitting
        self.train_gen, self.val_gen, self.test_gen = self.data_prep.create_data_generators(
            batch_size=32,
            seed=None,  # None for truly random split each time
            validation_split=0.2,  # 20% for validation
            use_merged=True
        )
        
        # Get class weights from training generator
        self.class_weights = self.data_prep.get_class_weights(
            self.train_gen,
            method='balanced'
        )
        
        # Get number of classes
        self.num_classes = len(self.train_gen.class_indices)
        self.class_names = list(self.train_gen.class_indices.keys())
        
        print(f"\nClasses: {self.class_names}")
        print(f"Training samples: {self.train_gen.samples}")
        print(f"Validation samples: {self.val_gen.samples}")
        print(f"Test samples: {self.test_gen.samples}")
        
        # Initialize model (will be created in build_model)
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build EfficientNet model with transfer learning."""
        print(f"\n{'='*80}")
        print(f"BUILDING MODEL")
        print(f"{'='*80}")
        
        # Load EfficientNet base model with ImageNet weights
        if self.model_variant == 'b2':
            base_model = keras.applications.EfficientNetB2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.input_size, self.input_size, 3)
            )
        elif self.model_variant == 'b3':
            base_model = keras.applications.EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=(self.input_size, self.input_size, 3)
            )
        elif self.model_variant == 'b4':
            base_model = keras.applications.EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_shape=(self.input_size, self.input_size, 3)
            )
        else:
            raise ValueError(f"Unsupported model variant: {self.model_variant}")
        
        # Freeze base model if not fine-tuning
        base_model.trainable = self.base_model_trainable
        
        if not self.base_model_trainable:
            print("✓ Base model frozen (transfer learning mode)")
        else:
            print("✓ Base model trainable (fine-tuning mode)")
        
        # Build model
        inputs = keras.Input(shape=(self.input_size, self.input_size, 3))
        
        # Preprocessing (EfficientNet preprocessing)
        x = keras.applications.efficientnet.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dropout for regularization
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Classification head
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        self.model = models.Model(inputs, outputs, name=f'EfficientNet-{self.model_variant.upper()}')
        
        print(f"\n✓ Model built successfully")
        print(f"  Total parameters: {self.model.count_params():,}")
        print(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        return self.model
    
    def compile_model(self, learning_rate=1e-4, optimizer='adam'):
        """
        Compile the model.
        
        Args:
            learning_rate (float): Initial learning rate
            optimizer (str): Optimizer name ('adam' or 'sgd')
        """
        if self.model is None:
            self.build_model()
        
        print(f"\n{'='*80}")
        print(f"COMPILING MODEL")
        print(f"{'='*80}")
        
        # Choose optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Custom metrics for comprehensive evaluation during training
        # Add precision, recall, and F1-score to monitor during training
        precision_metric = keras.metrics.Precision(name='precision')
        recall_metric = keras.metrics.Recall(name='recall')
        
        # F1-Score metric (custom implementation)
        class F1Score(keras.metrics.Metric):
            def __init__(self, name='f1_score', **kwargs):
                super().__init__(name=name, **kwargs)
                self.precision = keras.metrics.Precision()
                self.recall = keras.metrics.Recall()
            
            def update_state(self, y_true, y_pred, sample_weight=None):
                self.precision.update_state(y_true, y_pred, sample_weight)
                self.recall.update_state(y_true, y_pred, sample_weight)
            
            def result(self):
                p = self.precision.result()
                r = self.recall.result()
                return 2 * ((p * r) / (p + r + keras.backend.epsilon()))
            
            def reset_state(self):
                self.precision.reset_state()
                self.recall.reset_state()
        
        f1_metric = F1Score()
        
        # Compile with all metrics
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                precision_metric,
                recall_metric,
                f1_metric
            ]
        )
        
        print(f"✓ Optimizer: {optimizer}")
        print(f"✓ Learning rate: {learning_rate}")
        print(f"✓ Loss: categorical_crossentropy")
        print(f"✓ Metrics: accuracy, precision, recall, f1_score")
    
    def train(self, epochs=50, batch_size=32, learning_rate=1e-4, 
              optimizer='adam', patience=10, min_lr=1e-7):
        """
        Train the model.
        
        Args:
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size
            learning_rate (float): Initial learning rate
            optimizer (str): Optimizer name
            patience (int): Early stopping patience
            min_lr (float): Minimum learning rate
        """
        if self.model is None:
            self.build_model()
        
        if not self.model.built:
            self.compile_model(learning_rate=learning_rate, optimizer=optimizer)
        
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL")
        print(f"{'='*80}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Optimizer: {optimizer}")
        print(f"Early stopping patience: {patience}")
        
        # Setup callbacks
        callbacks = self._setup_callbacks(patience=patience, min_lr=min_lr)
        
        # Train model
        print(f"\nStarting training...")
        self.history = self.model.fit(
            self.train_gen,
            epochs=epochs,
            validation_data=self.val_gen,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Training completed!")
        return self.history
    
    def _setup_callbacks(self, patience=10, min_lr=1e-7):
        """Setup training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint (use .keras format to avoid HDF5 warning)
        checkpoint_path = self.models_dir / 'best_model.keras'
        model_checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=min_lr,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # CSV logger
        csv_logger = CSVLogger(
            filename=str(self.models_dir / 'training_log.csv'),
            append=False
        )
        callbacks.append(csv_logger)
        
        # TensorBoard (optional)
        tensorboard_dir = self.models_dir / 'tensorboard_logs'
        tensorboard = TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def evaluate(self):
        """Evaluate the model on test set."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL")
        print(f"{'='*80}")
        
        # Evaluate on test set
        test_results = self.model.evaluate(self.test_gen, verbose=1)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_results[0]:.4f}")
        print(f"  Accuracy: {test_results[1]:.4f}")
        if len(test_results) > 2:
            print(f"  Top-K Accuracy: {test_results[2]:.4f}")
        
        return test_results
    
    def save_model(self, save_path=None):
        """
        Save the trained model.
        
        Args:
            save_path (str): Optional custom save path
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if save_path is None:
            # Use .keras format to avoid HDF5 warning
            save_path = self.models_dir / 'final_model.keras'
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert .h5 to .keras if needed
            if save_path.suffix == '.h5':
                save_path = save_path.with_suffix('.keras')
        
        print(f"\n{'='*80}")
        print(f"SAVING MODEL")
        print(f"{'='*80}")
        
        # Save model (using .keras format - format determined by file extension)
        self.model.save(str(save_path))
        print(f"✓ Model saved to: {save_path}")
        
        # Save model configuration
        config = {
            'model_variant': self.model_variant,
            'version': self.version,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'class_indices': self.train_gen.class_indices,
            'base_model_trainable': self.base_model_trainable,
            'dropout_rate': self.dropout_rate,
            'training_samples': self.train_gen.samples,
            'validation_samples': self.val_gen.samples,
            'test_samples': self.test_gen.samples,
            'class_weights': {str(k): float(v) for k, v in self.class_weights.items()}
        }
        
        config_path = self.models_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to: {config_path}")
        
        # Save training history if available
        if self.history is not None:
            history_dict = {key: [float(val) for val in values] 
                           for key, values in self.history.history.items()}
            history_path = self.models_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            print(f"✓ Training history saved to: {history_path}")
            
            # Visualize training curves
            self.visualize_training_curves()
        
        return save_path
    
    def visualize_training_curves(self):
        """Visualize and save training curves."""
        if self.history is None or plt is None:
            return
        
        print(f"\n{'='*80}")
        print(f"VISUALIZING TRAINING CURVES")
        print(f"{'='*80}")
        
        assets_dir = Path('../docs/assets')
        assets_dir.mkdir(exist_ok=True)
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Curves - EfficientNet-{self.model_variant.upper()} (v{self.version})',
                     fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Precision, Recall, F1-Score curves (evaluation metrics)
        if 'precision' in history and 'recall' in history and 'f1_score' in history:
            axes[1, 0].plot(epochs, history['precision'], 'g-', label='Training Precision', linewidth=2)
            axes[1, 0].plot(epochs, history['recall'], 'orange', label='Training Recall', linewidth=2)
            axes[1, 0].plot(epochs, history['f1_score'], 'purple', label='Training F1-Score', linewidth=2)
            
            if 'val_precision' in history and 'val_recall' in history and 'val_f1_score' in history:
                axes[1, 0].plot(epochs, history['val_precision'], 'g--', label='Validation Precision', linewidth=2)
                axes[1, 0].plot(epochs, history['val_recall'], 'orange', linestyle='--', label='Validation Recall', linewidth=2)
                axes[1, 0].plot(epochs, history['val_f1_score'], 'purple', linestyle='--', label='Validation F1-Score', linewidth=2)
            
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Score', fontsize=12)
            axes[1, 0].set_title('Precision, Recall, F1-Score', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=9, loc='best')
            axes[1, 0].grid(alpha=0.3)
            axes[1, 0].set_ylim([0, 1])
        else:
            axes[1, 0].axis('off')
            axes[1, 0].text(0.5, 0.5, 'Precision/Recall/F1\nNot Available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        
        # Learning rate (if available)
        if 'lr' in history:
            axes[1, 1].plot(epochs, history['lr'], 'g-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(alpha=0.3)
        else:
            # Show model summary info
            axes[1, 1].axis('off')
            info_text = f"""Model: EfficientNet-{self.model_variant.upper()}
Version: {self.version}
Input Size: {self.input_size}×{self.input_size}
Classes: {self.num_classes}
Training Samples: {self.train_gen.samples}
Validation Samples: {self.val_gen.samples}"""
            axes[1, 1].text(0.5, 0.5, info_text, ha='center', va='center',
                          transform=axes[1, 1].transAxes, fontsize=11,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        curve_path = assets_dir / f'training_curves_{self.model_variant}_{self.version}.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved to: {curve_path}")
        plt.close()
        
        # Also save model architecture diagram
        if self.model is not None:
            try:
                from tensorflow.keras.utils import plot_model
                arch_path = assets_dir / f'model_architecture_{self.model_variant}_{self.version}.png'
                plot_model(self.model, to_file=str(arch_path), show_shapes=True, 
                          show_layer_names=True, rankdir='TB', dpi=150)
                print(f"✓ Model architecture saved to: {arch_path}")
            except Exception as e:
                print(f"⚠ Could not save model architecture: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train EfficientNet for Brain Tumor Detection')
    parser.add_argument('--model', type=str, default='b2', choices=['b2', 'b3', 'b4'],
                        help='EfficientNet variant (default: b2)')
    parser.add_argument('--version', type=str, default='v1.0',
                        help='Model version (default: v1.0)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer (default: adam)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tune base EfficientNet layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--dataset', type=str, default='../dataset',
                        help='Dataset path (default: ../dataset)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"BRAIN TUMOR DETECTION - EFFICIENTNET TRAINING")
    print(f"{'='*80}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nArguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = EfficientNetTrainer(
        model_variant=args.model,
        version=args.version,
        dataset_path=args.dataset,
        base_model_trainable=args.fine_tune,
        dropout_rate=args.dropout
    )
    
    # Build and compile model
    trainer.build_model()
    trainer.compile_model(learning_rate=args.learning_rate, optimizer=args.optimizer)
    
    # Train model
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        patience=args.patience
    )
    
    # Evaluate model
    trainer.evaluate()
    
    # Save model
    trainer.save_model()
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved to: {trainer.models_dir}")


if __name__ == '__main__':
    main()
