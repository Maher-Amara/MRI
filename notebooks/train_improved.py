"""
Improved Training Script for Brain Tumor Detection using EfficientNet
Addresses model plateau at 75% accuracy with comprehensive improvements:

1. Loss Functions: Focal Loss, Dice Loss, Tversky Loss
2. Metrics: Optimize loss only, monitor val_loss/AUC-ROC/AUC-PR, report others
3. Data Quality: Filter low-resolution scans
4. Optimization: Dynamic LR, early stopping, increased batch size
5. Best Practices: Avoid optimizing F1/accuracy directly

Usage:
    python train_improved.py --model b2 --version v2.0 --loss focal --batch-size 96
"""

from datapreparation import DataPreparation
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
from sklearn.metrics import roc_auc_score, average_precision_score

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*HDF5.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    plt = None
    print("Warning: Matplotlib not available. Training curves will not be visualized.")


# Set memory growth for GPU if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU available: {len(gpus)} device(s)")
except Exception as e:
    print(f"GPU configuration note: {e}")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def focal_loss(gamma=2.0, alpha=None):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Weighting factor for rare class (None for auto-balancing)
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions to avoid numerical instability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)

        # Calculate p_t (probability of true class)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        p_t = tf.clip_by_value(p_t, 1e-7, 1.0 - 1e-7)

        # Calculate focal weight
        focal_weight = tf.pow(1.0 - p_t, gamma)

        # Apply alpha weighting if provided
        if alpha is not None:
            alpha_t = tf.reduce_sum(alpha * y_true, axis=-1, keepdims=True)
            focal_weight = alpha_t * focal_weight

        # Calculate focal loss
        focal_loss = focal_weight * ce

        return tf.reduce_mean(focal_loss)

    return loss_fn


def dice_loss(smooth=1e-6):
    """
    Dice Loss for segmentation/classification tasks.
    Good for imbalanced datasets.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Dice Loss = 1 - Dice
    """
    def loss_fn(y_true, y_pred):
        # Flatten tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)

        # Calculate Dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)

        # Return Dice loss
        return 1.0 - dice

    return loss_fn


def tversky_loss(alpha=0.5, beta=0.5, smooth=1e-6):
    """
    Tversky Loss - generalization of Dice Loss.
    Allows control over false positives vs false negatives.

    Tversky = |A ∩ B| / (|A ∩ B| + alpha * |A - B| + beta * |B - A|)
    Tversky Loss = 1 - Tversky

    Args:
        alpha: Weight for false positives (default: 0.5)
        beta: Weight for false negatives (default: 0.5)
        smooth: Smoothing factor
    """
    def loss_fn(y_true, y_pred):
        # Flatten tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        # Calculate true positives, false positives, false negatives
        tp = tf.reduce_sum(y_true_flat * y_pred_flat)
        fp = tf.reduce_sum((1.0 - y_true_flat) * y_pred_flat)
        fn = tf.reduce_sum(y_true_flat * (1.0 - y_pred_flat))

        # Calculate Tversky coefficient
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

        # Return Tversky loss
        return 1.0 - tversky

    return loss_fn


def categorical_focal_loss(gamma=2.0, alpha=None):
    """
    Multi-class Focal Loss.
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross entropy
        ce = -y_true * tf.math.log(y_pred)

        # Probability of true class
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        p_t = tf.clip_by_value(p_t, 1e-7, 1.0 - 1e-7)

        # Focal weight
        focal_weight = tf.pow(1.0 - p_t, gamma)

        # Alpha weighting
        if alpha is not None:
            alpha_t = tf.reduce_sum(alpha * y_true, axis=-1, keepdims=True)
            focal_weight = alpha_t * focal_weight

        # Focal loss
        focal_loss = focal_weight * ce

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    return loss_fn


# ============================================================================
# METRICS (Monitoring Only - NOT for Optimization)
# ============================================================================

class AUCROC(keras.metrics.Metric):
    """AUC-ROC metric for monitoring (not optimization)."""

    def __init__(self, name='auc_roc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_labels = []
        self.pred_probs = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Store true labels and predictions
        y_true_np = tf.keras.backend.get_value(y_true)
        y_pred_np = tf.keras.backend.get_value(y_pred)

        # Convert to numpy if needed
        if isinstance(y_true_np, tf.Tensor):
            y_true_np = y_true_np.numpy()
        if isinstance(y_pred_np, tf.Tensor):
            y_pred_np = y_pred_np.numpy()

        self.true_labels.append(y_true_np)
        self.pred_probs.append(y_pred_np)

    def result(self):
        if len(self.true_labels) == 0:
            return 0.0

        # Concatenate all batches
        y_true_all = np.concatenate(self.true_labels, axis=0)
        y_pred_all = np.concatenate(self.pred_probs, axis=0)

        # Calculate AUC-ROC (one-vs-rest for multi-class)
        try:
            if y_true_all.shape[1] > 2:
                # Multi-class: use one-vs-rest
                auc = roc_auc_score(y_true_all, y_pred_all,
                                    average='macro', multi_class='ovr')
            else:
                # Binary
                auc = roc_auc_score(y_true_all[:, 1], y_pred_all[:, 1])
            return float(auc)
        except:
            return 0.0

    def reset_state(self):
        self.true_labels = []
        self.pred_probs = []


class AUCPR(keras.metrics.Metric):
    """AUC-PR (Average Precision) metric for monitoring."""

    def __init__(self, name='auc_pr', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_labels = []
        self.pred_probs = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_np = tf.keras.backend.get_value(y_true)
        y_pred_np = tf.keras.backend.get_value(y_pred)

        if isinstance(y_true_np, tf.Tensor):
            y_true_np = y_true_np.numpy()
        if isinstance(y_pred_np, tf.Tensor):
            y_pred_np = y_pred_np.numpy()

        self.true_labels.append(y_true_np)
        self.pred_probs.append(y_pred_np)

    def result(self):
        if len(self.true_labels) == 0:
            return 0.0

        y_true_all = np.concatenate(self.true_labels, axis=0)
        y_pred_all = np.concatenate(self.pred_probs, axis=0)

        try:
            if y_true_all.shape[1] > 2:
                # Multi-class: average precision for each class
                ap = average_precision_score(
                    y_true_all, y_pred_all, average='macro')
            else:
                ap = average_precision_score(
                    y_true_all[:, 1], y_pred_all[:, 1])
            return float(ap)
        except:
            return 0.0

    def reset_state(self):
        self.true_labels = []
        self.pred_probs = []


# ============================================================================
# IMPROVED TRAINER CLASS
# ============================================================================

class ImprovedEfficientNetTrainer:
    """Improved trainer with advanced loss functions and proper metrics usage."""

    EFFICIENTNET_VARIANTS = {
        'b2': {'size': 260, 'name': 'EfficientNetB2'},
        'b3': {'size': 300, 'name': 'EfficientNetB3'},
        'b4': {'size': 380, 'name': 'EfficientNetB4'}
    }

    LOSS_FUNCTIONS = {
        'categorical_crossentropy': 'categorical_crossentropy',  # String for Keras built-in
        'focal': lambda alpha=None: categorical_focal_loss(gamma=2.0, alpha=alpha),
        'dice': dice_loss,
        'tversky': lambda alpha=0.5, beta=0.5: tversky_loss(alpha=alpha, beta=beta),
    }

    def __init__(self, model_variant='b2', version='v2.0', dataset_path='../dataset',
                 base_model_trainable=False, dropout_rate=0.2, filter_low_quality=True):
        """Initialize the improved trainer."""
        if model_variant.lower() not in self.EFFICIENTNET_VARIANTS:
            raise ValueError(
                f"Model variant must be one of {list(self.EFFICIENTNET_VARIANTS.keys())}")

        self.model_variant = model_variant.lower()
        self.version = version
        self.dataset_path = Path(dataset_path)
        self.base_model_trainable = base_model_trainable
        self.dropout_rate = dropout_rate

        self.input_size = self.EFFICIENTNET_VARIANTS[self.model_variant]['size']
        self.model_name = self.EFFICIENTNET_VARIANTS[self.model_variant]['name']

        self.models_dir = Path('../models/efficientnet') / \
            version / f'efficientnet_{self.model_variant}'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data preparation (filtering is done during merge_datasets)
        print(f"\n{'='*80}")
        print(f"INITIALIZING IMPROVED TRAINER")
        print(f"{'='*80}")
        print(f"Model: EfficientNet-{self.model_variant.upper()}")
        print(f"Version: {version}")
        print(f"Input Size: {self.input_size}×{self.input_size}")
        print(f"Base Model Trainable: {base_model_trainable}")
        print(f"Data Filtering: {'Enabled' if filter_low_quality else 'Disabled'}")

        self.data_prep = DataPreparation(
            dataset_path=str(self.dataset_path),
            model_variant=self.model_variant,
            create_splits=False,
            filter_low_quality=filter_low_quality
        )

        self.model = None
        self.history = None

    def build_model(self):
        """Build EfficientNet model with transfer learning."""
        print(f"\n{'='*80}")
        print(f"BUILDING MODEL")
        print(f"{'='*80}")

        # Load EfficientNet base model
        if self.model_variant == 'b2':
            base_model = keras.applications.EfficientNetB2(
                weights='imagenet', include_top=False,
                input_shape=(self.input_size, self.input_size, 3)
            )
        elif self.model_variant == 'b3':
            base_model = keras.applications.EfficientNetB3(
                weights='imagenet', include_top=False,
                input_shape=(self.input_size, self.input_size, 3)
            )
        elif self.model_variant == 'b4':
            base_model = keras.applications.EfficientNetB4(
                weights='imagenet', include_top=False,
                input_shape=(self.input_size, self.input_size, 3)
            )

        base_model.trainable = self.base_model_trainable

        if not self.base_model_trainable:
            print("✓ Base model frozen (transfer learning mode)")
        else:
            print("✓ Base model trainable (fine-tuning mode)")

        # Build model
        inputs = keras.Input(shape=(self.input_size, self.input_size, 3))
        x = keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Get num_classes from train_gen (will be set later)
        num_classes = 4  # Default
        outputs = layers.Dense(
            num_classes, activation='softmax', name='predictions')(x)

        self.model = models.Model(
            inputs, outputs, name=f'EfficientNet-{self.model_variant.upper()}')

        print(f"\n✓ Model built successfully")
        print(f"  Total parameters: {self.model.count_params():,}")
        print(
            f"  Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")

        return self.model

    def compile_model(self, train_gen, loss_type='focal',
                      learning_rate=1e-4, optimizer='adam', loss_params=None):
        """
        Compile model with proper metrics usage.

        CRITICAL: Only optimize loss. Metrics are for monitoring/reporting only.
        """
        if self.model is None:
            self.build_model()

        # Update output layer for correct number of classes
        num_classes = len(train_gen.class_indices)
        # Check model output shape (need to build model first to get output_shape)
        if not self.model.built:
            # Build model with dummy input to get output shape
            dummy_input = tf.zeros((1, self.input_size, self.input_size, 3))
            _ = self.model(dummy_input)
        
        current_output_shape = self.model.output_shape
        if current_output_shape[-1] != num_classes:
            x = self.model.layers[-2].output
            outputs = layers.Dense(
                num_classes, activation='softmax', name='predictions')(x)
            self.model = models.Model(self.model.input, outputs)

        print(f"\n{'='*80}")
        print(f"COMPILING MODEL")
        print(f"{'='*80}")

        # Choose optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Get loss function
        if loss_type not in self.LOSS_FUNCTIONS:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Choose from {list(self.LOSS_FUNCTIONS.keys())}")

        loss_fn = self.LOSS_FUNCTIONS[loss_type]

        if loss_type == 'categorical_crossentropy':
            # String loss function (Keras built-in)
            loss = loss_fn
        else:
            # Custom loss function - all return a callable loss function
            if loss_params:
                loss = loss_fn(**loss_params)
            else:
                loss = loss_fn()  # Call the function/lambda to get the actual loss function

        # METRICS: Only for monitoring, NOT for optimization
        # - Optimize: Loss only
        # - Monitor: val_loss, AUC-ROC, AUC-PR
        # - Report: Accuracy, Recall, F1, Confusion Matrix, Specificity

        # Note: We don't add accuracy/precision/recall/F1 to metrics during training
        # because we should NOT optimize on these. We'll calculate them during evaluation.

        # For monitoring during training, we can add AUC metrics (but they're expensive)
        # For now, we'll keep it simple and only track loss

        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=[]  # No metrics during training - we optimize loss only
        )

        print(f"✓ Optimizer: {optimizer}")
        print(f"✓ Learning rate: {learning_rate}")
        print(f"✓ Loss: {loss_type}")
        print(f"✓ Metrics: None (optimizing loss only)")
        print(f"\n⚠ IMPORTANT: Optimizing loss only. Metrics calculated during evaluation.")

    def train(self, epochs=50, batch_size=32, learning_rate=1e-4,
              optimizer='adam', patience=10, min_lr=1e-7,
              loss_type='focal', loss_params=None):
        """Train the model."""
        if self.model is None:
            self.build_model()

        if not self.model.built:
            # Create data generators first
            train_gen, val_gen, test_gen = self.data_prep.create_data_generators(
                batch_size=batch_size,
                seed=42,
                validation_split=0.2,
                use_merged=True
            )

            class_weights = self.data_prep.get_class_weights(
                train_gen, method='balanced')
            self.train_gen = train_gen
            self.val_gen = val_gen
            self.test_gen = test_gen
            self.class_weights = class_weights

            self.compile_model(train_gen, loss_type,
                               learning_rate, optimizer, loss_params)

        print(f"\n{'='*80}")
        print(f"TRAINING MODEL")
        print(f"{'='*80}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Optimizer: {optimizer}")
        print(f"Loss: {loss_type}")
        print(f"Early stopping patience: {patience}")
        print(f"Monitor: val_loss (optimizing loss only)")

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

        # Early stopping on val_loss
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Model checkpoint on val_loss
        checkpoint_path = self.models_dir / 'best_model.keras'
        model_checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)

        # Dynamic learning rate reduction on val_loss
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

        # TensorBoard
        tensorboard_dir = self.models_dir / 'tensorboard_logs'
        tensorboard = TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1
        )
        callbacks.append(tensorboard)

        return callbacks

    def evaluate(self):
        """Evaluate model and calculate all metrics for reporting."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL")
        print(f"{'='*80}")

        # Get predictions
        print("Generating predictions...")
        y_true = []
        y_pred = []

        self.test_gen.reset()
        for i in range(len(self.test_gen)):
            batch_x, batch_y = next(self.test_gen)
            batch_pred = self.model.predict(batch_x, verbose=0)
            y_true.append(batch_y)
            y_pred.append(batch_pred)

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report,
            roc_auc_score, average_precision_score
        )

        # Overall metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision = precision_score(
            y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(
            y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

        # AUC-ROC (one-vs-rest for multi-class)
        try:
            auc_roc = roc_auc_score(
                y_true, y_pred, average='macro', multi_class='ovr')
        except:
            auc_roc = 0.0

        # AUC-PR (Average Precision)
        try:
            auc_pr = average_precision_score(y_true, y_pred, average='macro')
        except:
            auc_pr = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # Per-class metrics
        precision_per_class = precision_score(
            y_true_classes, y_pred_classes, average=None)
        recall_per_class = recall_score(
            y_true_classes, y_pred_classes, average=None)
        f1_per_class = f1_score(y_true_classes, y_pred_classes, average=None)

        # Specificity (True Negative Rate) per class
        specificity_per_class = []
        for i in range(len(self.test_gen.class_indices)):
            tn = np.sum((y_true_classes != i) & (y_pred_classes != i))
            fp = np.sum((y_true_classes != i) & (y_pred_classes == i))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_per_class.append(specificity)

        # Print results
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc_roc:.4f}")
        print(f"  AUC-PR:    {auc_pr:.4f}")

        print(f"\nPer-Class Metrics:")
        class_names = list(self.test_gen.class_indices.keys())
        print(
            f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Specificity':<12}")
        print("-" * 80)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<25} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
                  f"{f1_per_class[i]:<12.4f} {specificity_per_class[i]:<12.4f}")

        print(f"\nConfusion Matrix:")
        print(cm)

        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'specificity': float(specificity_per_class[i])
                }
                for i in range(len(class_names))
            }
        }

        results_path = self.models_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_path}")

        return results

    def save_model(self):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        save_path = self.models_dir / 'final_model.keras'
        self.model.save(str(save_path))
        print(f"✓ Model saved to: {save_path}")

        # Save config
        config = {
            'model_variant': self.model_variant,
            'version': self.version,
            'input_size': self.input_size,
            'base_model_trainable': self.base_model_trainable,
            'dropout_rate': self.dropout_rate
        }

        config_path = self.models_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save history
        if self.history is not None:
            history_dict = {key: [float(val) for val in values]
                            for key, values in self.history.history.items()}
            history_path = self.models_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)

            self.visualize_training_curves()

        return save_path

    def visualize_training_curves(self):
        """Visualize and save training curves."""
        if self.history is None or plt is None:
            return

        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Training Curves - EfficientNet-{self.model_variant.upper()} (v{self.version})',
                     fontsize=16, fontweight='bold')

        # Loss
        axes[0].plot(epochs, history['loss'], 'b-',
                     label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(epochs, history['val_loss'], 'r-',
                         label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Model Loss (Optimized)',
                          fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Learning rate
        if 'lr' in history:
            axes[1].plot(epochs, history['lr'], 'g-', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Learning Rate', fontsize=12)
            axes[1].set_title('Learning Rate Schedule',
                              fontsize=14, fontweight='bold')
            axes[1].set_yscale('log')
            axes[1].grid(alpha=0.3)
        else:
            axes[1].axis('off')

        plt.tight_layout()
        assets_dir = Path('../docs/assets')
        assets_dir.mkdir(exist_ok=True)
        curve_path = assets_dir / \
            f'training_curves_{self.model_variant}_{self.version}.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved to: {curve_path}")
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Improved Training for Brain Tumor Detection')
    parser.add_argument('--model', type=str, default='b2', choices=['b2', 'b3', 'b4'],
                        help='EfficientNet variant (default: b2)')
    parser.add_argument('--version', type=str, default='v2.0',
                        help='Model version (default: v2.0 for improved training script)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32, try 64, 96, or 128 if memory allows)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer (default: adam)')
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['categorical_crossentropy',
                                 'focal', 'dice', 'tversky'],
                        help='Loss function (default: focal - recommended for imbalanced medical imaging)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tune base EfficientNet layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--dataset', type=str, default='../dataset',
                        help='Dataset path (default: ../dataset)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip low-quality image filtering')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"IMPROVED BRAIN TUMOR DETECTION TRAINING")
    print(f"{'='*80}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Initialize trainer
    trainer = ImprovedEfficientNetTrainer(
        model_variant=args.model,
        version=args.version,
        dataset_path=args.dataset,
        base_model_trainable=args.fine_tune,
        dropout_rate=args.dropout,
        filter_low_quality=not args.no_filter
    )

    # Create data generators
    train_gen, val_gen, test_gen = trainer.data_prep.create_data_generators(
        batch_size=args.batch_size,
        seed=42,
        validation_split=0.2,
        use_merged=True
    )

    class_weights = trainer.data_prep.get_class_weights(
        train_gen, method='balanced')
    trainer.train_gen = train_gen
    trainer.val_gen = val_gen
    trainer.test_gen = test_gen
    trainer.class_weights = class_weights

    # Build and compile model
    trainer.build_model()
    trainer.compile_model(
        train_gen,
        loss_type=args.loss,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer
    )

    # Train model
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        patience=args.patience,
        loss_type=args.loss
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
