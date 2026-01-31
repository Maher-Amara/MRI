# Design Documentation

## 1. Overview

This document describes the design of the Brain Tumor Detection (BTD) system, including machine learning models and deep learning architectures. All design choices are justified and explained in detail.

## 2. Design Approach

The BTD system can be implemented using two complementary approaches:

### 2.1 Machine Learning (ML) Approach

- Traditional feature extraction methods
- Classical ML classifiers (SVM, Random Forest, etc.)
- Hand-crafted features from MRI images

### 2.2 Deep Learning (DL) Approach

- Convolutional Neural Networks (CNNs)
- Transfer learning from pre-trained models
- Custom architectures designed for medical imaging

## 3. Recommended Architecture: Deep Learning

### 3.1 Why Deep Learning?

- **Automatic Feature Extraction**: CNNs automatically learn relevant features from images
- **Proven Performance**: State-of-the-art results in medical image classification
- **Scalability**: Can handle large datasets effectively
- **Transfer Learning**: Leverage pre-trained models for better performance

### 3.1.1 Dataset-Driven Model Choice

Based on the current dataset analysis:

- **Dataset size**: 3,264 images across 4 classes (moderate size)
- **Format**: RGB `.jpg`, mostly `512x512` with size variance
- **Class balance**: Mild imbalance (no_tumor is ~15.3% of the dataset)

**Implications for model choice** (based on research):

- Use **transfer learning** to reduce overfitting on a moderate-sized dataset
- **Selected 3 candidate models**: EfficientNet B2 (primary), B3 (alternative), B4 (experimental)
  - **B2 (260×260)**: **PRIMARY** - Recommended for ~2,500 images, achieves 98-99.5% accuracy on similar MRI datasets
  - **B3 (300×300)**: **ALTERNATIVE** - Good balance, slightly larger than recommended
  - **B4 (380×380)**: **EXPERIMENTAL** - Recommended for 10k+ images, may risk overfitting on 3,264 images
- **Research guidance**:
  - B0: Best for ~400 images (too small for our 3,264)
  - B2: Optimal for ~2,500 images (matches our dataset size closely)
  - B4: For 10k+ images (may be too large, requires strong regularization)
- **Excluded**: B0/B1 (too small), B5 (similar to B4), B6/B7 (exceed 512×512, would require upscaling)
- All selected models fit within image dimensions without upscaling
- Apply **class weights** or targeted augmentation for the smaller no_tumor class

### 3.2 Architecture Options

#### Option 1: Custom CNN Architecture

```txt
Input Layer (MRI Image)
    ↓
Convolutional Block 1
    - Conv2D (32 filters, 3x3)
    - Batch Normalization
    - ReLU Activation
    - MaxPooling2D (2x2)
    ↓
Convolutional Block 2
    - Conv2D (64 filters, 3x3)
    - Batch Normalization
    - ReLU Activation
    - MaxPooling2D (2x2)
    ↓
Convolutional Block 3
    - Conv2D (128 filters, 3x3)
    - Batch Normalization
    - ReLU Activation
    - MaxPooling2D (2x2)
    ↓
Convolutional Block 4
    - Conv2D (256 filters, 3x3)
    - Batch Normalization
    - ReLU Activation
    - MaxPooling2D (2x2)
    ↓
Flatten Layer
    ↓
Dense Layer 1 (512 units)
    - Dropout (0.5)
    - ReLU Activation
    ↓
Dense Layer 2 (256 units)
    - Dropout (0.5)
    - ReLU Activation
    ↓
Output Layer
    - Dense (num_classes)
    - Softmax Activation (for multi-class)
    - Sigmoid Activation (for binary)
```

#### Option 2: Transfer Learning

**Pre-trained Models to Consider:**

- **ResNet50/ResNet101**: Deep residual networks with skip connections
- **VGG16/VGG19**: Simple and effective architecture
- **EfficientNet**: Efficient and accurate models
- **DenseNet**: Dense connections between layers
- **MobileNet**: Lightweight for faster inference

**Transfer Learning Pipeline:**

```txt
Pre-trained Model (ImageNet weights)
    ↓
Remove Top Layers
    ↓
Add Custom Classification Head
    - Global Average Pooling
    - Dense Layer (512 units)
    - Dropout (0.5)
    - Output Layer (num_classes)
```

### 3.3 Layer-by-Layer Explanation

#### Convolutional Layers

- **Purpose**: Extract spatial features from images
- **Filters**: Learn edge, texture, and pattern detectors
- **Kernel Size**: 3x3 is standard for medical imaging
- **Padding**: 'same' to preserve spatial dimensions

#### Batch Normalization

- **Purpose**: Stabilize training and accelerate convergence
- **Benefits**: Reduces internal covariate shift, allows higher learning rates

#### Activation Functions

- **ReLU**: Non-linear activation, prevents vanishing gradients
- **Softmax**: Multi-class classification output
- **Sigmoid**: Binary classification output

#### Pooling Layers

- **Purpose**: Reduce spatial dimensions, control overfitting
- **Type**: MaxPooling preserves important features

#### Dropout

- **Purpose**: Regularization to prevent overfitting
- **Rate**: 0.5 is common for dense layers

## 4. Model Comparison Strategy

### 4.1 Models to Compare

**Selected EfficientNet Models** (based on dataset constraints and research):

1. **EfficientNet B2 (260×260)**: **PRIMARY** - Recommended for ~2,500 images, 98-99.5% accuracy on MRI
2. **EfficientNet B3 (300×300)**: **ALTERNATIVE** - Good balance, slightly larger than recommended
3. **EfficientNet B4 (380×380)**: **EXPERIMENTAL** - For 10k+ images, may risk overfitting on 3,264

**Selection Rationale** (Research-Based):

- Dataset: 3,264 images (between 2,500 and 10k range)
- Image dimensions: Mean ~467×470, most common 512×512
- **B0**: Best for ~400 images (too small for our 3,264)
- **B2**: Optimal for ~2,500 images (matches our dataset size closely) ✓
- **B4**: Recommended for 10k+ images (may be too large, requires strong regularization)
- **B6/B7**: Exceed 512×512, would require upscaling (not desired)
- All selected models fit within image dimensions without upscaling

**Alternative Models** (for comparison):
4. **Custom CNN**: Baseline architecture
5. **ResNet50**: Transfer learning baseline
6. **VGG16**: Alternative transfer learning

### 4.2 Comparison Metrics

- Training accuracy and loss
- Validation accuracy and loss
- Test set performance
- Inference time
- Model size
- Training time

## 5. Parameter Tuning

### 5.1 Hyperparameters to Tune

#### Learning Rate

- **Range**: 1e-5 to 1e-2
- **Method**: Learning rate scheduling or adaptive optimizers
- **Recommendation**: Start with 1e-4, use ReduceLROnPlateau

#### Batch Size

- **Range**: 16, 32, 64, 128
- **Consideration**: GPU memory constraints
- **Recommendation**: 32 for most cases

#### Epochs

- **Range**: 20-100
- **Method**: Early stopping to prevent overfitting
- **Recommendation**: Monitor validation loss

#### Optimizer

- **Options**: Adam, SGD, RMSprop
- **Recommendation**: Adam with default parameters

#### Regularization

- **Dropout**: 0.3-0.7 (higher for B4 to prevent overfitting on smaller dataset)
- **L2 Regularization**: 1e-4 to 1e-6
- **Data Augmentation**: Rotation, flipping, zooming (critical for 3,264 images)
- **Early Stopping**: Monitor validation loss to prevent overfitting
- **Class Weights**: Apply to handle class imbalance (no_tumor is smaller)
- **Note**: B4 may require stronger regularization (higher dropout, lower LR) due to dataset size

### 5.2 Tuning Strategy

1. **Grid Search**: Systematic exploration of hyperparameter space
2. **Random Search**: More efficient for high-dimensional spaces
3. **Bayesian Optimization**: Efficient for expensive evaluations
4. **Manual Tuning**: Based on validation performance

## 6. Preprocessing Pipeline

### 6.1 Image Preprocessing Steps

#### EfficientNet Input Size Requirements

EfficientNet models require **square input images**, and the size depends on the model variant. Based on the [Keras EfficientNet fine-tuning guide](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/), the standard input sizes are:

| Model Variant      | Input Size | Parameters | ImageNet Top-1 | MRI Acc.* | Status        | Notes                                    |
| ------------------ | ---------- | ---------- | -------------- | --------- | ------------- | ---------------------------------------- |
| EfficientNet-B0    | 224×224    | 5.3M       | 77.1%          | 97-99%    | Excluded      | Best for ~400 images (too small)         |
| EfficientNet-B1    | 240×240    | 7.8M       | 79.1%          | -         | Excluded      | Similar to B2 but less optimal          |
| **EfficientNet-B2** | **260×260** | **9.2M**   | **80.1%**      | **98-99.5%** | **PRIMARY**    | Recommended for ~2,500 images            |
| **EfficientNet-B3** | **300×300** | **12M**    | **81.6%**      | **98.5%+**  | **Alternative** | Good balance, slightly larger            |
| **EfficientNet-B4** | **380×380** | **19M**    | **83.0%**      | **98.5%+**  | **Experimental** | For 10k+ images, may overfit on 3,264    |
| EfficientNet-B5    | 456×456    | 30M        | 83.7%          | -         | Excluded      | Similar to B4, less optimal              |
| EfficientNet-B6    | 528×528    | 43M        | 84.1%          | -         | Excluded      | Exceeds 512×512, requires upscaling      |
| EfficientNet-B7    | 600×600    | 66M        | 84.4%          | -         | Excluded      | Exceeds 512×512, requires upscaling      |

\* *Typical accuracy on similar 4-class brain MRI tumor datasets*

**Recommendation**:

- **Start with EfficientNet-B2** (PRIMARY) - optimally sized for our dataset (3,264 images), achieves 98-99.5% accuracy on similar MRI brain tumor classification tasks
- **Test EfficientNet-B3** (ALTERNATIVE) - good balance, slightly larger than recommended
- **Experiment with EfficientNet-B4** (EXPERIMENTAL) - may require stronger regularization to prevent overfitting

**Training Tips for B4** (if experimenting):

- Use lower initial learning rate (1e-5 instead of 1e-4)
- Apply higher dropout (0.5-0.7)
- Use early stopping with patience
- Monitor validation loss closely for overfitting signs

**Preprocessing Steps:**

1. **Resizing**: Resize images to match the EfficientNet variant's input size
   - Use `tf.keras.preprocessing.image.smart_resize()` or similar
   - Maintain aspect ratio using padding (not cropping) to avoid losing image data
   - For our dataset (mostly 512×512), resize to target size with padding if needed

2. **Normalization**: Scale pixel values to [0, 1] range
   - EfficientNet expects pixel values in [0, 255] range, then normalized
   - Use `tf.keras.applications.efficientnet.preprocess_input()` for proper normalization

3. **Color Handling**: Keep RGB format (dataset is already RGB)
   - No conversion needed as all images are RGB

4. **Aspect Ratio Preservation**:
   - Use padding (not cropping) to maintain square format
   - **Black padding is used** - this is appropriate for MRI images because:
     - MRI scan borders are naturally black
     - Skull appears grey in MRI scans
     - Black padding matches the natural appearance and doesn't introduce artifacts
   - This ensures no tumor information is lost during preprocessing

**Visualization**:

![Augmentation Examples](assets/augmentation_examples_b2.png)
*Data augmentation examples showing geometric transformations (rotation, translation, zoom out, flip) with original MRI colors preserved*

### 6.2 Data Augmentation Strategy

**Critical Principle**: All augmentations must preserve the full image content. We use **zoom out** and **translation** to avoid accidentally cropping important tumor regions.

#### Safe Augmentation Techniques (Geometric Only)

**Key Principle**: Original MRI colors are NEVER changed. Only geometric transformations are applied with black padding.

1. **Zoom Out (Not Zoom In)**:
   - Range: 0.95-1.0 (zoom out only, never zoom in, very minimal to prevent cropping)
   - Ensures we never crop the image
   - Adds black padding around the image when zooming out
   - Original image colors preserved

2. **Translation (Safe Amounts)**:
   - Horizontal/vertical shift: ±2% of image size (very conservative to prevent cropping)
   - Use `fill_mode='constant'` with `cval=0.0` (black padding)
   - Very small translations preserve all image content
   - Original image colors preserved

3. **Rotation (Safe Angles)**:
   - Range: ±3 degrees maximum (very conservative to prevent cropping)
   - Use black padding (`fill_mode='constant'`, `cval=0.0`) to avoid cropping corners
   - Very small rotations maintain full image visibility
   - Original image colors preserved

4. **Horizontal Flip**:
   - Safe for medical images (brain symmetry)
   - No data loss, just mirroring
   - Preserves original colors

**Important: NO Color Adjustments**

- **NO brightness adjustment** - Original MRI colors must be preserved
- **NO contrast adjustment** - Color changes don't reflect real-world MRI scans
- **Only geometric transformations** - Rotation, translation, zoom out, flip
- **Only black padding** - Added during transformations, matches natural MRI borders

#### Augmentation Implementation

```python
# Example: Safe augmentation pipeline
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    # Zoom out only (never zoom in to avoid cropping)
    zoom_range=[0.85, 1.0],  # Zoom out 15% max
    
    # Small translations with padding
    width_shift_range=0.1,   # ±10% horizontal shift
    height_shift_range=0.1,  # ±10% vertical shift
    fill_mode='constant',    # Pad with constant value
    
    # Safe rotation with padding
    rotation_range=15,       # ±15 degrees
    fill_mode='constant',   # Pad rotated corners
    
    # Horizontal flip (safe)
    horizontal_flip=True,
    
    # Brightness/contrast (conservative)
    brightness_range=[0.85, 1.15],  # ±15% brightness
    
    # Normalization
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)
```

### 6.3 Justification

- **Model-Specific Resizing**: EfficientNet requires square inputs of specific sizes. Using the correct size for each variant ensures optimal performance.
- **Padding Over Cropping**: Medical images contain critical information at all edges. Padding preserves all data, while cropping risks losing tumor regions.
- **Black Padding**: Black padding matches natural MRI appearance (borders are black, skull is grey). Original image colors are preserved.
- **Zoom Out Strategy**: Zooming out adds context without losing information, while zooming in would crop important regions. Only black padding is added.
- **Color Preservation**: Original MRI image colors/intensities are preserved. Only geometric transformations (rotation, translation, zoom) are applied. No brightness/contrast adjustments that would alter diagnostic quality.
- **Conservative Augmentation**: Medical imaging requires careful augmentation. Small, safe geometric transformations preserve diagnostic quality while still providing regularization.
- **Normalization**: Proper EfficientNet preprocessing is applied during training (not during visualization) to ensure compatibility with ImageNet-pretrained weights.

## 7. Training Strategy

**Visualizations**:

![Training Curves](assets/training_curves_b2_v1.0.png)
*Training curves showing model performance over epochs (loss, accuracy, learning rate)*

![Model Architecture](assets/model_architecture_b2_v1.0.png)
*EfficientNet model architecture showing layer structure and connections*

### 7.1 Data Splitting

**Decision: Merge Datasets and Use Random 80/20 Split**

- **Original split**: 2,870 / 394 (87.9% / 12.1%) - **Merged due to class disparities**
- **New strategy**: Random 80% train / 20% validation split during training
  - Datasets merged to address class disparities between original Training/Testing folders
  - Random split each training run (seed=None) prevents overfitting to specific distribution
  - Better class balance and more reliable evaluation
- **Original Testing folder**: Can still be used as separate test set if needed
- **Class weights**: Applied during training to handle class imbalance

### 7.2 Training Process

1. **Initialization**: Pre-trained weights or random initialization
2. **Freezing**: Optionally freeze early layers in transfer learning
3. **Fine-tuning**: Gradually unfreeze layers
4. **Monitoring**: Track training and validation metrics
5. **Early Stopping**: Stop when validation loss plateaus

### 7.3 Loss Functions

- **Binary Classification**: Binary Cross-Entropy
- **Multi-class Classification**: Categorical Cross-Entropy
- **Class Imbalance**: Weighted loss or focal loss

## 8. Implementation Considerations

### 8.1 Platform Selection

- **Google Colab**: Free GPU access, easy sharing
- **Kaggle**: Competition environment, datasets available
- **Local Setup**: More control, requires GPU

### 8.2 Memory Management

- **Batch Size**: Adjust based on available memory
- **Image Size**: Balance between resolution and memory
- **Gradient Accumulation**: For effective larger batch sizes

### 8.3 Visualization

- **Layer Outputs**: Visualize feature maps from each layer
- **Grad-CAM**: Visualize which regions contribute to predictions
- **Training Curves**: Plot loss and accuracy over epochs

## 9. Model Selection Criteria

1. **Performance**: Highest accuracy, precision, recall, F1-score
2. **Efficiency**: Fast inference time for clinical use
3. **Robustness**: Consistent performance across different cases
4. **Interpretability**: Ability to explain predictions
5. **Simplicity**: Balance between complexity and performance

## 10. Future Improvements

- **Ensemble Methods**: Combine multiple models
- **Attention Mechanisms**: Focus on relevant image regions
- **3D CNNs**: Process volumetric MRI data
- **Multi-modal Learning**: Combine different MRI sequences
- **Active Learning**: Improve with minimal labeled data
