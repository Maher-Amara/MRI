# Model B2 Improvements Guide

## Overview

This guide documents the comprehensive improvements made to address the model plateau at 75% accuracy. The improved training script (`train_improved.py`) implements best practices for medical image classification.

## Key Improvements

### 1. Loss Functions

**✅ RECOMMENDED: Focal Loss (Default)**

For brain tumor detection with class imbalance, **Focal Loss is the best choice** and is now the default.

**Why Focal Loss for Our Use Case:**

- ✅ **Addresses Class Imbalance**: Our dataset has imbalanced classes (no_tumor: 15.3%, others: ~28%)
- ✅ **Focuses on Hard Examples**: Emphasizes difficult-to-classify cases, which is critical for medical imaging
- ✅ **Proven in Medical Imaging**: Widely used and effective for medical classification tasks
- ✅ **Better than Standard Cross-Entropy**: Reduces weight of easy examples, preventing model from being dominated by majority classes
- ✅ **Formula**: `FL = -alpha * (1 - p_t)^gamma * log(p_t)` where gamma=2.0 focuses on hard examples

**Other Loss Functions (Available but Not Recommended):**

- **Categorical Cross-Entropy**: Standard loss, but doesn't handle imbalance well
- **Dice Loss**: Better for segmentation tasks, less effective for classification
- **Tversky Loss**: Useful when you need to tune FP/FN ratio, but adds complexity

**Usage:**

```bash
# Use recommended Focal Loss (now default)
python train_improved.py --model b2

# Or explicitly specify (same as default)
python train_improved.py --loss focal --model b2

# Use other loss functions if needed (not recommended)
python train_improved.py --loss categorical_crossentropy --model b2
```

### 2. Metrics (Correct Usage)

**CRITICAL PRINCIPLE: Optimize loss only, use metrics for monitoring/reporting**

#### During Training

- **Optimize**: Loss only (categorical_crossentropy, focal, dice, or tversky)
- **Monitor**:
  - `val_loss` (for early stopping and LR reduction)
  - AUC-ROC (calculated during evaluation)
  - AUC-PR (calculated during evaluation)

#### During Evaluation (Reporting)

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (one-vs-rest for multi-class)
- **AUC-PR**: Area under Precision-Recall curve
- **Confusion Matrix**: Per-class performance
- **Specificity**: True negatives / (True negatives + False positives)

**Why This Matters:**

- ❌ **WRONG**: Optimizing F1-score or accuracy directly can lead to poor generalization
- ✅ **CORRECT**: Optimize loss, which provides smooth gradients and better optimization
- Metrics are calculated after training for comprehensive evaluation

### 3. Optimization Improvements

#### Dynamic Learning Rate

- **ReduceLROnPlateau**: Automatically reduces learning rate when `val_loss` plateaus
- Factor: 0.5 (halves learning rate)
- Patience: `patience // 2` (half of early stopping patience)
- Minimum LR: 1e-7

#### Early Stopping

- **Monitor**: `val_loss` (not accuracy!)
- **Patience**: 10 epochs (configurable)
- **Restore Best Weights**: Automatically restores best model weights

#### Batch Size

- **Default**: 32
- **Recommended**: Increase to 64, 96, or 128 if memory allows
- **Benefits**:
  - More stable gradients
  - Better generalization
  - Faster training (fewer iterations per epoch)

**Usage:**

```bash
# Increase batch size (if GPU memory allows)
python train_improved.py --batch-size 96 --model b2

# Try tripling the batch size
python train_improved.py --batch-size 96 --model b2
```

### 4. Data Quality Filtering

**✅ RECOMMENDED: Enable Filtering (Default)**

**Filtering is enabled by default and strongly recommended** for medical imaging applications.

**What Filtering Does:**

- Removes images with resolution below model's input size
  - B2: Removes images < 260×260
  - B3: Removes images < 300×300
  - B4: Removes images < 380×380
- Removes corrupted/unreadable images
- Improves overall dataset quality

**Why Filtering is Critical for Our Use Case:**

- ✅ **Medical Imaging Quality Matters**: Low-resolution scans provide poor diagnostic signal
- ✅ **Prevents Learning from Noise**: Prevents model from learning patterns from low-quality data
- ✅ **Improves Generalization**: Better quality data leads to better model performance
- ✅ **Standard Practice**: Medical imaging pipelines typically include quality checks
- ✅ **Minimal Data Loss**: Typically removes <5% of images, but significantly improves quality

**Usage:**

```bash
# Use recommended filtering (default - recommended)
python train_improved.py --model b2

# Skip filtering (NOT recommended - only for testing)
python train_improved.py --model b2 --no-filter
```

**Recommendation: Always use filtering for production models.**

### 5. Common Traps Avoided

#### ❌ Traps to Avoid

1. **Optimizing F1/Accuracy Directly**
   - These metrics are not differentiable
   - Can lead to poor generalization
   - Solution: Optimize loss, report metrics

2. **Trusting Batch-Wise Metrics**
   - Batch metrics can be noisy
   - Solution: Use validation set metrics

3. **Training Longer Without Changing Objective**
   - If plateaued, need to change approach
   - Solution: Try different loss functions, increase batch size, filter data

4. **Using Top-K Accuracy for Binary Tasks**
   - Not applicable for binary/multi-class classification
   - Solution: Use standard accuracy, precision, recall, F1

#### ✅ Best Practices Implemented

- Optimize loss only
- Monitor validation loss
- Early stopping on validation loss
- Dynamic learning rate reduction
- Comprehensive evaluation metrics
- Data quality filtering

## Usage Examples

### Example 1: Recommended Configuration (Default Settings)

**This is the recommended setup for brain tumor detection:**

```bash
python train_improved.py \
    --model b2 \
    --version v2.0 \
    --loss focal \
    --batch-size 64 \
    --epochs 50 \
    --patience 10
```

**Why this configuration:**

- ✅ Focal Loss: Best for imbalanced medical imaging
- ✅ Batch size 64: Good balance (increase to 96-128 if memory allows)
- ✅ Filtering enabled: Ensures data quality
- ✅ Standard epochs/patience: Prevents overfitting

### Example 2: Maximum Performance (If GPU Memory Allows)

```bash
python train_improved.py \
    --model b2 \
    --version v2.0 \
    --loss focal \
    --batch-size 96 \
    --epochs 50 \
    --patience 10
```

**Benefits:**

- Larger batch size = more stable gradients
- Faster training (fewer iterations per epoch)
- Better generalization

### Example 3: Fine-Tuning for Final Push

```bash
python train_improved.py \
    --model b2 \
    --version v2.1 \
    --loss focal \
    --batch-size 64 \
    --fine-tune \
    --learning-rate 1e-5 \
    --epochs 100 \
    --patience 15
```

**When to use:**

- After initial training with frozen base model
- Lower learning rate for fine-tuning
- More patience for longer training

## Expected Improvements

With these improvements, you should see:

1. **Better Convergence**: Loss functions designed for imbalanced data
2. **Higher Accuracy**: Proper optimization strategy
3. **Better Generalization**: Data quality filtering and proper metrics
4. **Faster Training**: Larger batch sizes (if memory allows)
5. **More Reliable Results**: Early stopping and dynamic LR

## Monitoring Training

### Key Metrics to Watch

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease and stabilize (monitored for early stopping)
3. **Learning Rate**: Should decrease when validation loss plateaus

### After Training

Check `evaluation_results.json` for:

- Overall accuracy, precision, recall, F1
- AUC-ROC and AUC-PR
- Per-class metrics
- Confusion matrix
- Specificity per class

## Troubleshooting

### If accuracy still plateaus

1. Try different loss function (focal, dice, tversky)
2. Increase batch size (if memory allows)
3. Enable fine-tuning: `--fine-tune`
4. Lower learning rate: `--learning-rate 1e-5`
5. Check data quality (ensure filtering is working)

### If out of memory

1. Reduce batch size: `--batch-size 32`
2. Use smaller model: `--model b2` instead of `b3` or `b4`
3. Disable fine-tuning (keeps base model frozen)

## File Structure

```
notebooks/
├── train_improved.py          # Improved training script
├── train.py                   # Original training script
├── datapreparation.py         # Data preparation (unchanged)
└── IMPROVEMENTS_GUIDE.md     # This file

models/efficientnet/
└── v2.0/
    └── efficientnet_b2/
        ├── best_model.keras           # Best model (lowest val_loss)
        ├── final_model.keras          # Final model after training
        ├── model_config.json          # Model configuration
        ├── training_history.json      # Training history
        ├── training_log.csv          # CSV log
        └── evaluation_results.json    # Comprehensive evaluation metrics
```

## Recommended Configuration Summary

**For Brain Tumor Detection, use these defaults:**

| Setting              | Recommended Value                    | Why                                      |
| -------------------- | ------------------------------------ | ---------------------------------------- |
| **Loss Function**    | `focal` (default)                    | Best for imbalanced medical imaging      |
| **Data Filtering**   | `enabled` (default)                   | Critical for medical image quality       |
| **Batch Size**        | `64` (or 96-128 if memory allows)     | Balance between stability and speed      |
| **Learning Rate**     | `1e-4` (default)                      | Good starting point, reduces automatically |
| **Early Stopping**    | `patience=10` (default)                | Prevents overfitting                     |
| **Model Variant**     | `b2` (default)                        | Optimal for ~3,264 images               |

**Quick Start Command:**

```bash
python train_improved.py --model b2 --batch-size 64
```

This uses all recommended defaults (Focal Loss + Filtering enabled).

## Next Steps

1. **Start with Recommended Configuration**: Use Focal Loss + Filtering (defaults)
2. **Increase Batch Size**: Try 64, 96, or 128 if GPU memory allows
3. **Monitor Validation Loss**: Ensure it's decreasing steadily
4. **Check Evaluation Results**: Review comprehensive metrics in `evaluation_results.json`
5. **Fine-Tune if Needed**: If plateau persists, try fine-tuning with `--fine-tune`

## References

- Focal Loss: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- Dice Loss: [Milletari et al., 2016](https://arxiv.org/abs/1606.04797)
- Tversky Loss: [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)
- Best Practices: Medical Image Classification Guidelines
