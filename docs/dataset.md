# Dataset Documentation

## Overview

This document provides information about the datasets used for the Brain Tumor Detection (BTD) system. The project requires labeled MRI brain scan images with multiple tumor types for training and evaluation.

## Dataset Requirements

### Characteristics

- **Type**: Labeled MRI brain scan images
- **Format**: JPEG, PNG, or DICOM
- **Resolution**: Variable (will be standardized during preprocessing)
- **Classes**:
  - Glioma tumors
  - Meningioma tumors
  - Pituitary tumors
  - No tumor (normal scans)

### Requirements

- **Diversity**:
  - Different patients
  - Various ages
  - Both genders
- **Size**: Sufficient for training, validation, and testing
- **Quality**: High-quality scans with clear tumor boundaries when present
- **Labels**: Accurate ground truth labels for all images

## Recommended Datasets

### 1. Brain Tumor Classification (MRI) - Kaggle

**Source**: [Kaggle - Brain Tumor Classification](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)

**Description**:

- Contains MRI images of brain tumors
- Multiple tumor types: Glioma, Meningioma, Pituitary
- Well-organized folder structure
- Suitable for classification tasks

**Characteristics**:

- Format: JPEG images
- Organization: Separate folders for each class
- Size: Typically 3000+ images
- Resolution: Variable

### 2. Brain MRI Images for Brain Tumor Detection

**Source**: [Kaggle - Brain MRI Images](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

**Description**:

- Binary classification dataset (Tumor / No Tumor)
- Good for initial detection tasks
- Can be combined with multi-class datasets

**Characteristics**:

- Format: JPEG/PNG
- Binary labels: Yes/No tumor
- Size: Variable

## Dataset Structure

### Original Directory Structure

```txt
dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

**Important Decision**: The original split (87.9% train / 12.1% test) has significant issues:
- **Class disparities**: Uneven class distribution between Training and Testing folders
- **Too small test set**: Only 12.1% (394 images) is insufficient for reliable evaluation
- **Solution**: Merge both folders and use random 80/20 train/validation split during training

### Data Preparation Process

1. **Merge Datasets**: Combine `Training/` and `Testing/` into a unified dataset
   - Addresses class disparities between original splits
   - Provides better control over data distribution
2. **Random Splitting**: Use 80% train / 20% validation split during training
   - Random split each training run (seed=None for true randomness)
   - Prevents overfitting to specific data distribution
   - More reliable model evaluation
3. **Class Imbalance Handling**: Apply class weights during training to prevent bias toward majority classes

### Final Directory Structure (After Preparation)

```txt
dataset/
├── Training/          # Original (kept for reference)
├── Testing/           # Original (kept for reference, can be used as separate test set)
└── merged/            # Merged dataset (all images combined)
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

**Note**: Random 80/20 split is performed during training, not stored in folders.

**Benefits of Merging and Random Splitting**:

- **Better Class Balance**: Random splitting ensures balanced class distribution (no class disparities)
- **Appropriate Validation Size**: 20% validation set (vs 12.1% original test set)
- **Randomness**: Different split each training run prevents overfitting to specific distribution
- **Flexibility**: Can adjust split ratio as needed (currently 80/20)
- **More Reliable Evaluation**: Proper validation set size for model assessment

## Data Collection

### Steps

1. **Download Dataset**
   - From Kaggle or other medical imaging repositories
   - Ensure proper licensing and usage rights
   - Verify data quality

2. **Organize Data**
   - Create train/validation/test splits
   - Maintain class distribution (stratified split)
   - Organize by class folders

3. **Validate Data**
   - Check for corrupted images
   - Verify labels are correct
   - Remove duplicates if present

## Data Preprocessing

### Notes from Current Dataset

- All images are RGB `.jpg` files
- Most images are `512x512`, but sizes vary; resize to EfficientNet-specific sizes
- **Class imbalance exists**:
  - Glioma: 28.37%, Meningioma: 28.71%, Pituitary: 27.60%, No Tumor: 15.32%
  - **Solution**: Use stratified splitting + class weights to prevent bias

### Standard Operations

#### EfficientNet-Specific Preprocessing

1. **Resizing to Model-Specific Square Sizes**:
   - **Selected Models** (based on dataset constraints and research):
     - **EfficientNet B2: 260×260 pixels** (PRIMARY - recommended for ~2,500 images)
     - **EfficientNet B3: 300×300 pixels** (ALTERNATIVE - good balance)
     - **EfficientNet B4: 380×380 pixels** (EXPERIMENTAL - for 10k+ images, may overfit)
   - **Selection Rationale** (Research-Based):
     - Dataset: 3,264 images (between 2,500 and 10k range)
     - Image dimensions: Mean ~467×470, most common 512×512
     - **B0**: Best for ~400 images (too small for our 3,264)
     - **B2**: Optimal for ~2,500 images (matches our dataset size closely) ✓
     - **B4**: Recommended for 10k+ images (may be too large, requires strong regularization)
     - **B6/B7**: Exceed 512×512, would require upscaling (not desired)
     - All selected models fit within 512×512 without upscaling
   - **Important**: Use black padding (not cropping) to preserve all image data

2. **Normalization**:
   - Use `tf.keras.applications.efficientnet.preprocess_input()` for proper normalization
   - Scales pixel values appropriately for EfficientNet models

3. **Format**: Keep RGB format (dataset is already RGB, no conversion needed)

4. **Aspect Ratio**:
   - Maintain square format using center padding with **black** (value 0.0)
   - Black padding is appropriate for MRI because:
     - MRI scan borders are naturally black
     - Skull appears grey in MRI scans
     - Black padding matches natural appearance without introducing artifacts
   - Never crop images to avoid losing tumor information

### Data Augmentation Strategy

**Key Principle**: All augmentations must preserve full image content. We use **zoom out** and safe **translation/rotation** to avoid accidentally cropping important regions.

#### Safe Augmentation Techniques

1. **Zoom Out Only** (0.95-1.0 range, very minimal to prevent cropping):
   - Zoom out to add context, never zoom in to avoid cropping
   - Ensures all original image content is preserved

2. **Translation** (±2% of image size, very conservative to prevent cropping):
   - Very small horizontal/vertical shifts with padding
   - Preserves all image regions

3. **Rotation** (±3 degrees maximum, very conservative to prevent cropping):
   - Very small rotations with padding to avoid corner cropping
   - Maintains full image visibility

4. **Horizontal Flip**:
   - Safe for brain MRI images (symmetrical anatomy)
   - No data loss
   - Preserves original colors

**Important: NO Color Adjustments**

- **NO brightness adjustment** - Original MRI colors must be preserved
- **NO contrast adjustment** - Color changes don't reflect real-world MRI scans
- **Only geometric transformations** - Rotation, translation, zoom out, flip
- **Only black padding** - Added during transformations, matches natural MRI borders

**Reference**: Based on [Keras EfficientNet fine-tuning guide](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)

**Visualization**:

![Augmentation Examples](assets/augmentation_examples_b2.png)
*Data augmentation examples showing safe geometric transformations that preserve original MRI colors and full image content*

## Dataset Statistics

### Current Dataset Summary (from Data Understanding Analysis)

The following visualizations were generated from the data understanding analysis:

![Sample Images](assets/sample_images.png)
*Sample MRI images from each tumor class*

![Class Distribution](assets/class_distribution.png)
*Distribution of images across classes and splits*

![Image Dimensions](assets/image_dimensions.png)
*Analysis of image dimensions showing width/height distributions*

![File Properties](assets/file_properties.png)
*File format, size, and channel analysis*

- **Total images**: 3,264
- **Original split**: 2,870 / 394 (87.9% / 12.1%) - **Merged due to class disparities**
- **New split strategy**: Random 80% train / 20% validation (during training)
- **Image format**: `.jpg` only
- **Color channels**: RGB only
- **Image dimensions**:
  - Min: `174x167`
  - Max: `1375x1446`
  - Mean: `467x470` (Std: `133x125`)
  - Most common: `512x512` (2,341 images)
- **File sizes**:
  - Min: `0.005 MB`
  - Max: `0.270 MB`
  - Mean: `0.027 MB`
  - Median: `0.025 MB`
  - Total size: `88.77 MB`
- **Corrupted images**: 0 detected

### Class Distribution

| Class | Training | Testing | Total | Share |
| ----- | -------- | ------- | ----- | ----- |
| glioma_tumor | 826 | 100 | 926 | 28.37% |
| meningioma_tumor | 822 | 115 | 937 | 28.71% |
| pituitary_tumor | 827 | 74 | 901 | 27.60% |
| no_tumor | 395 | 105 | 500 | 15.32% |
| **Total** | **2,870** | **394** | **3,264** | **100%** |

## Data Quality Checks

### Validation Steps

1. **Image Quality**
   - Check for corrupted files
   - Verify readable formats
   - Assess image clarity

2. **Label Accuracy**
   - Spot-check label correctness
   - Verify class assignments
   - Check for mislabeled data

3. **Distribution**
   - Check for class imbalance
   - Verify train/val/test splits maintain distribution
   - Identify potential biases

4. **Completeness**
   - Ensure all images have labels
   - Check for missing files
   - Verify directory structure

## Ethical Considerations

### Data Usage

- **Privacy**: Ensure patient privacy is maintained
- **Consent**: Use only publicly available or properly consented data
- **Attribution**: Properly cite data sources
- **Purpose**: Use data only for research/educational purposes

### Medical Disclaimer

- This system is for research/educational purposes
- Not intended for clinical diagnosis without proper validation
- Always consult medical professionals for actual diagnoses

## References

### Dataset Sources

1. **Brain Tumor Classification (MRI)**
   - URL: <https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri>
   - Description: Multi-class brain tumor classification dataset

2. **Brain MRI Images for Brain Tumor Detection**
   - URL: <https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection>
   - Description: Binary classification dataset

### Implementation References

1. **PyTorch Brain Tumor Detector**
   - URL: <https://github.com/MLDawn/MLDawn-Projects/blob/main/Pytorch/Brain-Tumor-Detector/MRI-Brain-Tumor-Detecor.ipynb>
   - Description: Example implementation using PyTorch

### Research Papers

1. **Nature Scientific Reports**
   - URL: <https://www.nature.com/articles/s41598-025-92776-1>
   - Description: Research paper on brain tumor detection methods

## Additional Resources

### Other Potential Datasets

- **BraTS (Brain Tumor Segmentation)**: 3D MRI volumes with segmentation masks
- **TCIA (The Cancer Imaging Archive)**: Large collection of medical images
- **Medical Image Datasets**: Various repositories for medical imaging

### Data Augmentation Tools

- TensorFlow/Keras ImageDataGenerator
- PyTorch torchvision.transforms
- Albumentations library
- imgaug library

## Notes

- Always verify dataset licenses before use
- Document any modifications made to original datasets
- Keep track of dataset versions used
- Maintain reproducibility by documenting exact dataset sources and versions
