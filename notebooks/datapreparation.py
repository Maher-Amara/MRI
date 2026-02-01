"""
Data Preparation Script for Brain Tumor Detection Project
Phase 3 of CRISP-DM Methodology

This script:
1. Merges Training and Testing datasets into a unified dataset
2. Creates stratified train/validation/test splits (maintaining class distribution)
3. Implements EfficientNet-specific preprocessing
4. Applies safe data augmentation that preserves all image content
5. Handles class imbalance with class weights and stratified splitting

Based on: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set memory growth for GPU if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"GPU configuration note: {e}")


class DataPreparation:
    """Class to prepare and preprocess MRI brain tumor dataset for EfficientNet models."""

    # EfficientNet input sizes (square images required)
    # Selected models based on dataset constraints and research:
    # - Dataset: 3,264 images (between 2,500 and 10k range)
    # - Image dimensions: Mean ~467×470, most common 512×512
    # - Research guidance:
    #   * B0: Best for ~400 images (too small for our 3,264)
    #   * B2: Recommended for ~2,500 images (matches our dataset size closely)
    #   * B4: Recommended for 10k+ images (too large for our 3,264, risks overfitting)
    # - Image size constraint: No upscaling (input size must be ≤ 512×512)
    # Selected candidates: B2 (primary), B3 (alternative), B4 (experimental)
    EFFICIENTNET_SIZES = {
        'b2': 260,  # PRIMARY: Recommended for ~2,500 images, 98-99.5% accuracy on MRI
        'b3': 300,  # ALTERNATIVE: Good balance, slightly larger than recommended
        'b4': 380   # EXPERIMENTAL: For 10k+ images, may overfit on 3,264 images
    }

    def __init__(self, dataset_path='../dataset', model_variant='b2', 
                 merged_data_path=None, create_splits=True,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 filter_low_quality=True):
        """
        Initialize the DataPreparation class.

        Args:
            dataset_path (str): Path to the dataset directory with Training/Testing folders
            model_variant (str): EfficientNet variant (b2, b3, or b4) to determine input size
            merged_data_path (str): Path where merged dataset will be stored (default: dataset_path/merged)
            create_splits (bool): Whether to create train/val/test splits
            train_ratio (float): Proportion for training set (default: 0.7)
            val_ratio (float): Proportion for validation set (default: 0.15)
            test_ratio (float): Proportion for test set (default: 0.15)
            filter_low_quality (bool): Filter low-resolution images before merging (default: True)
        """
        self.dataset_path = Path(dataset_path)
        self.training_path = self.dataset_path / 'Training'
        self.testing_path = self.dataset_path / 'Testing'
        self.model_variant = model_variant.lower()
        
        # Validate ratios sum to 1.0
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.create_splits = create_splits
        self.filter_low_quality = filter_low_quality

        # Get input size for the model variant
        if self.model_variant not in self.EFFICIENTNET_SIZES:
            raise ValueError(
                f"Invalid model variant: {model_variant}. Choose from {list(self.EFFICIENTNET_SIZES.keys())}")

        self.input_size = self.EFFICIENTNET_SIZES[self.model_variant]
        
        # Set merged data path
        if merged_data_path is None:
            self.merged_data_path = self.dataset_path / 'merged'
        else:
            self.merged_data_path = Path(merged_data_path)
        
        print(f"Using EfficientNet-{self.model_variant.upper()} with input size: {self.input_size}×{self.input_size}")
        print(f"Train/Val/Test split: {train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%}")

    def preprocess_image(self, image_path, target_size=None):
        """
        Preprocess a single image for EfficientNet.

        Args:
            image_path: Path to the image file
            target_size: Target size (default: model's input size)

        Returns:
            Preprocessed image array
        """
        if target_size is None:
            target_size = (self.input_size, self.input_size)

        # Load image
        img = keras.utils.load_img(image_path)
        img_array = keras.utils.img_to_array(img)

        # Resize with padding to preserve aspect ratio (no cropping)
        # Use smart_resize or manual padding
        img_array = self._resize_with_padding(img_array, target_size)

        # Expand dimensions for batch
        img_array = np.expand_dims(img_array, axis=0)

        # Apply EfficientNet preprocessing
        img_array = keras.applications.efficientnet.preprocess_input(img_array)

        return img_array[0]  # Remove batch dimension

    def _resize_with_padding(self, img_array, target_size):
        """
        Resize image with padding to maintain aspect ratio (no cropping).

        Args:
            img_array: Image array
            target_size: Target (width, height) - must be square for EfficientNet

        Returns:
            Resized image with padding
        """
        target_w, target_h = target_size
        img_h, img_w = img_array.shape[:2]

        # Calculate scaling factor to fit image in target size
        scale = min(target_w / img_w, target_h / img_h)

        # Resize image
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_resized = tf.image.resize(
            img_array, (new_h, new_w), method='bilinear')

        # Calculate padding
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2

        # Pad image with black (appropriate for MRI - borders are naturally black, skull is grey)
        # Black padding matches the natural appearance of MRI scan borders
        img_padded = tf.image.pad_to_bounding_box(
            img_resized,
            offset_height=pad_h,
            offset_width=pad_w,
            target_height=target_h,
            target_width=target_w
        )

        return img_padded.numpy()
    
    def filter_low_quality_images(self, image_files, min_resolution=None):
        """
        Filter out low-quality images (low resolution or corrupted).
        
        Args:
            image_files: List of image file paths to check
            min_resolution: Minimum resolution (default: model input size)
        
        Returns:
            Tuple of (filtered_image_files, removed_count)
        """
        if min_resolution is None:
            min_resolution = self.input_size
        
        filtered_files = []
        removed_count = 0
        
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    width, height = img.size
                    min_dim = min(width, height)
                    
                    if min_dim >= min_resolution:
                        filtered_files.append(img_file)
                    else:
                        removed_count += 1
            except Exception:
                # Corrupted image
                removed_count += 1
        
        return filtered_files, removed_count
    
    def merge_datasets(self, force_merge=False):
        """
        Merge Training and Testing folders into a unified dataset.
        
        Args:
            force_merge (bool): If True, overwrite existing merged dataset
        
        Returns:
            Path to merged dataset directory
        """
        print("\n" + "=" * 80)
        print("MERGING TRAINING AND TESTING DATASETS")
        print("=" * 80)
        
        # Check if merged dataset already exists
        if self.merged_data_path.exists() and not force_merge:
            print(f"✓ Merged dataset already exists at: {self.merged_data_path}")
            print("  Use force_merge=True to overwrite")
            return self.merged_data_path
        
        # Filter low-quality images if enabled
        if self.filter_low_quality:
            print(f"\n{'='*80}")
            print(f"FILTERING LOW-QUALITY IMAGES (Before Merging)")
            print(f"{'='*80}")
            print(f"Minimum resolution: {self.input_size}×{self.input_size}")
        
        # Create merged directory
        self.merged_data_path.mkdir(parents=True, exist_ok=True)
        
        # Get all class names
        expected_classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        
        # Merge images from both Training and Testing
        class_counts = defaultdict(int)
        total_checked = 0
        total_removed = 0
        
        for split_name, split_path in [('Training', self.training_path), ('Testing', self.testing_path)]:
            if not split_path.exists():
                print(f"⚠ Warning: {split_path} does not exist, skipping...")
                continue
            
            print(f"\nProcessing {split_name}...")
            for class_name in expected_classes:
                class_path = split_path / class_name
                if not class_path.exists():
                    continue
                
                # Create class directory in merged dataset
                merged_class_path = self.merged_data_path / class_name
                merged_class_path.mkdir(exist_ok=True)
                
                # Get all images
                image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
                total_checked += len(image_files)
                
                # Filter low-quality images if enabled
                if self.filter_low_quality:
                    filtered_files, removed = self.filter_low_quality_images(image_files, self.input_size)
                    total_removed += removed
                    image_files = filtered_files
                
                # Copy filtered images
                for img_file in image_files:
                    # Create unique filename to avoid conflicts
                    new_name = f"{split_name.lower()}_{img_file.name}"
                    dest_path = merged_class_path / new_name
                    shutil.copy2(img_file, dest_path)
                    class_counts[class_name] += 1
                
                if self.filter_low_quality:
                    print(f"  {class_name}: {len(image_files)} images copied (filtered from {len(image_files) + removed})")
                else:
                    print(f"  {class_name}: {len(image_files)} images copied")
        
        if self.filter_low_quality:
            print(f"\nFiltering Summary:")
            print(f"  Checked: {total_checked} images")
            print(f"  Removed: {total_removed} low-quality/corrupted images")
            print(f"  Kept: {total_checked - total_removed} images")
        
        print("\n" + "=" * 80)
        print("MERGE SUMMARY")
        print("=" * 80)
        total = 0
        for class_name in expected_classes:
            count = class_counts[class_name]
            total += count
            print(f"  {class_name}: {count} images")
        print(f"  TOTAL: {total} images")
        print(f"\n✓ Merged dataset created at: {self.merged_data_path}")
        
        return self.merged_data_path
    
    def create_stratified_splits(self, seed=42, force_split=False):
        """
        Create stratified train/validation/test splits from merged dataset.
        Maintains class distribution in each split to handle class imbalance.
        
        Args:
            seed (int): Random seed for reproducibility
            force_split (bool): If True, overwrite existing splits
        
        Returns:
            Tuple of (train_path, val_path, test_path)
        """
        print("\n" + "=" * 80)
        print("CREATING STRATIFIED TRAIN/VALIDATION/TEST SPLITS")
        print("=" * 80)
        
        # Check if merged dataset exists
        if not self.merged_data_path.exists():
            print("⚠ Merged dataset not found. Merging datasets first...")
            self.merge_datasets()
        
        # Create split directories
        splits_dir = self.dataset_path / 'splits'
        train_path = splits_dir / 'train'
        val_path = splits_dir / 'validation'
        test_path = splits_dir / 'test'
        
        # Check if splits already exist
        if train_path.exists() and val_path.exists() and test_path.exists() and not force_split:
            print(f"✓ Splits already exist at: {splits_dir}")
            print("  Use force_split=True to recreate")
            return train_path, val_path, test_path
        
        # Create split directories
        for split_path in [train_path, val_path, test_path]:
            split_path.mkdir(parents=True, exist_ok=True)
        
        # Get all class names
        expected_classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        
        # Collect all images per class
        class_images = defaultdict(list)
        for class_name in expected_classes:
            class_path = self.merged_data_path / class_name
            if class_path.exists():
                image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
                class_images[class_name] = image_files
        
        print("\nCreating stratified splits (maintaining class distribution)...")
        
        # Create splits for each class
        split_counts = defaultdict(lambda: defaultdict(int))
        
        for class_name, image_files in class_images.items():
            if len(image_files) == 0:
                continue
            
            # Shuffle for randomness
            np.random.seed(seed)
            np.random.shuffle(image_files)
            
            # Calculate split sizes
            n_total = len(image_files)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            n_test = n_total - n_train - n_val  # Remaining goes to test
            
            # Split images
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]
            
            # Create class directories in split folders
            for split_path, files in [(train_path, train_files), 
                                      (val_path, val_files), 
                                      (test_path, test_files)]:
                class_split_path = split_path / class_name
                class_split_path.mkdir(exist_ok=True)
                
                # Copy files
                for img_file in files:
                    dest_path = class_split_path / img_file.name
                    shutil.copy2(img_file, dest_path)
                    split_counts[split_path.name][class_name] += 1
        
        # Print summary
        print("\n" + "=" * 80)
        print("SPLIT SUMMARY")
        print("=" * 80)
        print(f"{'Class':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
        print("-" * 80)
        
        for class_name in expected_classes:
            train_count = split_counts['train'][class_name]
            val_count = split_counts['validation'][class_name]
            test_count = split_counts['test'][class_name]
            total = train_count + val_count + test_count
            
            if total > 0:
                print(f"{class_name:<25} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")
        
        # Print totals
        train_total = sum(split_counts['train'].values())
        val_total = sum(split_counts['validation'].values())
        test_total = sum(split_counts['test'].values())
        grand_total = train_total + val_total + test_total
        
        print("-" * 80)
        print(f"{'TOTAL':<25} {train_total:<10} {val_total:<10} {test_total:<10} {grand_total:<10}")
        print(f"\nSplit ratios: Train={train_total/grand_total:.1%}, Val={val_total/grand_total:.1%}, Test={test_total/grand_total:.1%}")
        
        # Check class distribution in each split
        print("\nClass Distribution Check:")
        for split_name in ['train', 'validation', 'test']:
            counts = split_counts[split_name]
            total = sum(counts.values())
            if total > 0:
                print(f"\n{split_name.upper()}:")
                for class_name in expected_classes:
                    count = counts[class_name]
                    if total > 0:
                        pct = (count / total) * 100
                        print(f"  {class_name}: {count} ({pct:.1f}%)")
        
        print(f"\n✓ Splits created at: {splits_dir}")
        
        return train_path, val_path, test_path

    def create_train_datagen(self, validation_split=0.2, for_visualization=False):
        """
        Create training data generator with safe augmentation.

        Key principle: All augmentations preserve full image content and original colors.
        - Zoom out only (never zoom in to avoid cropping)
        - Small translations with black padding
        - Safe rotation angles with black padding
        - Original colors preserved (only padding is black)

        Args:
            validation_split: Fraction of data to use for validation
            for_visualization: If True, skip preprocessing to preserve original colors

        Returns:
            ImageDataGenerator for training
        """
        # Geometric augmentations ONLY - preserve original MRI colors
        # Key principle: Only geometric transformations (rotation, translation, zoom)
        # Original image colors are NEVER changed - only black padding is added
        # Very conservative ranges to completely prevent any cropping
        train_datagen = ImageDataGenerator(
            # Zoom out only (0.95-1.0) - very minimal zoom to prevent cropping
            zoom_range=[0.95, 1.0],

            # Very small translations with black padding to prevent cropping
            width_shift_range=0.02,   # ±2% horizontal shift (very conservative)
            height_shift_range=0.02,  # ±2% vertical shift (very conservative)
            fill_mode='constant',     # Pad with black (natural for MRI borders)
            cval=0.0,                 # Black padding value (matches MRI scan borders)

            # Very small rotation with black padding to prevent cropping
            rotation_range=3,         # ±3 degrees maximum (very conservative)
            # Note: fill_mode and cval already set above for translations

            # Horizontal flip (safe for brain MRI, preserves colors)
            horizontal_flip=True,
            vertical_flip=False,      # Not recommended for brain images

            # NO brightness/contrast/color adjustments
            # Original MRI colors must be preserved - brightness changes don't reflect real-world scans
            # brightness_range, contrast adjustments, etc. are NOT used

            # EfficientNet preprocessing (only for training, not for visualization)
            # This normalizes pixel values but doesn't change the relative color relationships
            preprocessing_function=keras.applications.efficientnet.preprocess_input if not for_visualization else None,

            # Validation split
            validation_split=validation_split
        )

        return train_datagen

    def create_test_datagen(self):
        """
        Create test data generator (no augmentation, only preprocessing).

        Returns:
            ImageDataGenerator for testing
        """
        test_datagen = ImageDataGenerator(
            preprocessing_function=keras.applications.efficientnet.preprocess_input
        )

        return test_datagen

    def create_data_generators(self, batch_size=32, seed=42, validation_split=0.2, use_merged=True):
        """
        Create train, validation, and test data generators with random splitting during training.
        Uses validation_split parameter for random train/val split (keeps it random each time).

        Args:
            batch_size: Batch size for training
            seed: Random seed for reproducibility (None for truly random)
            validation_split: Fraction of data to use for validation (default: 0.2)
            use_merged: If True, use merged dataset; if False, use original Training/Testing

        Returns:
            Tuple of (train_gen, val_gen, test_gen)
        """
        # Merge datasets if needed
        if use_merged:
            merged_path = self.merged_data_path
            if not merged_path.exists() or not any(merged_path.iterdir()):
                print("Merging datasets...")
                self.merge_datasets()
            data_path = merged_path
        else:
            # Use original Training folder for train/val, Testing for test
            data_path = self.training_path
        
        # Training generator with augmentation and random validation split
        train_datagen = self.create_train_datagen(validation_split=validation_split, for_visualization=False)
        
        train_gen = train_datagen.flow_from_directory(
            data_path,
            target_size=(self.input_size, self.input_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            seed=seed,
            shuffle=True
        )

        # Validation generator (no augmentation, same directory with validation subset)
        val_datagen = self.create_test_datagen()
        val_datagen_with_split = ImageDataGenerator(
            preprocessing_function=keras.applications.efficientnet.preprocess_input,
            validation_split=validation_split
        )
        
        val_gen = val_datagen_with_split.flow_from_directory(
            data_path,
            target_size=(self.input_size, self.input_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            seed=seed,
            shuffle=False
        )

        # Test generator (use Testing folder if available, otherwise use validation)
        test_datagen = self.create_test_datagen()
        
        if self.testing_path.exists() and any(self.testing_path.iterdir()):
            test_gen = test_datagen.flow_from_directory(
                self.testing_path,
                target_size=(self.input_size, self.input_size),
                batch_size=batch_size,
                class_mode='categorical',
                seed=seed,
                shuffle=False
            )
        else:
            # No separate test set, use validation as test
            print("⚠ Warning: No separate test set found. Using validation set as test.")
            test_gen = val_gen

        return train_gen, val_gen, test_gen

    def get_class_weights(self, train_gen, method='balanced'):
        """
        Calculate class weights to handle class imbalance.
        
        Methods:
        - 'balanced': sklearn-style balanced weights (n_samples / (n_classes * count))
        - 'inverse': Simple inverse frequency (1 / count)
        - 'sqrt': Square root of inverse frequency (1 / sqrt(count))

        Args:
            train_gen: Training data generator
            method: Method for calculating weights ('balanced', 'inverse', 'sqrt')

        Returns:
            Dictionary of class weights
        """
        class_counts = train_gen.classes
        total_samples = len(class_counts)
        num_classes = len(train_gen.class_indices)

        # Count samples per class
        class_counts_dict = {}
        for class_idx in range(num_classes):
            class_counts_dict[class_idx] = np.sum(class_counts == class_idx)

        # Calculate weights based on method
        class_weights = {}
        
        if method == 'balanced':
            # sklearn-style balanced weights
            for class_idx, count in class_counts_dict.items():
                if count > 0:
                    class_weights[class_idx] = total_samples / (num_classes * count)
                else:
                    class_weights[class_idx] = 0.0
        elif method == 'inverse':
            # Simple inverse frequency
            max_count = max(class_counts_dict.values())
            for class_idx, count in class_counts_dict.items():
                if count > 0:
                    class_weights[class_idx] = max_count / count
                else:
                    class_weights[class_idx] = 0.0
        elif method == 'sqrt':
            # Square root of inverse frequency (less aggressive)
            max_count = max(class_counts_dict.values())
            for class_idx, count in class_counts_dict.items():
                if count > 0:
                    class_weights[class_idx] = np.sqrt(max_count / count)
                else:
                    class_weights[class_idx] = 0.0
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'balanced', 'inverse', 'sqrt'")

        print(f"\nClass Weights (method: {method}):")
        print("-" * 80)
        print(f"{'Class':<25} {'Count':<10} {'Weight':<10} {'% of Total':<10}")
        print("-" * 80)
        
        for class_name, class_idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
            count = class_counts_dict[class_idx]
            weight = class_weights[class_idx]
            pct = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"{class_name:<25} {count:<10} {weight:<10.3f} {pct:<10.1f}%")
        
        print("-" * 80)
        print(f"{'TOTAL':<25} {total_samples:<10}")
        
        # Calculate imbalance ratio
        counts = list(class_counts_dict.values())
        if counts:
            max_count = max(counts)
            min_count = min([c for c in counts if c > 0])
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}")
            if imbalance_ratio > 2.0:
                print("⚠ Significant class imbalance detected. Class weights will help balance training.")
            else:
                print("✓ Classes are relatively balanced.")

        return class_weights

    def visualize_augmentation(self, num_samples=8):
        """
        Visualize augmented images to verify augmentation preserves content and original colors.

        Args:
            num_samples: Number of augmented samples to show
        """
        # Get a sample image from merged dataset
        if (self.dataset_path / 'merged').exists():
            sample_class_path = self.dataset_path / 'merged'
        else:
            sample_class_path = self.training_path
        
        sample_class = list(sample_class_path.iterdir())[0]
        sample_image = list(sample_class.iterdir())[0]

        # Load original image
        img = keras.utils.load_img(sample_image)
        img_array = keras.utils.img_to_array(img)
        
        # Resize to model input size with padding (preserve aspect ratio)
        img_array = self._resize_with_padding(img_array, (self.input_size, self.input_size))
        img_array = np.expand_dims(img_array, axis=0)

        # Create augmentation generator WITHOUT preprocessing to preserve original colors
        datagen = self.create_train_datagen(for_visualization=True)

        # Generate augmented images
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Data Augmentation Examples (EfficientNet-{self.model_variant.upper()}, {self.input_size}×{self.input_size})\nOriginal Colors Preserved - Only Geometric Transformations',
                     fontsize=14, fontweight='bold')

        axes = axes.flatten()

        # Original image (normalized to 0-1 for display)
        original_display = img_array[0] / 255.0
        axes[0].imshow(original_display)
        axes[0].set_title('Original Image', fontsize=10)
        axes[0].axis('off')

        # Augmented images (preserve original colors, no preprocessing)
        aug_iter = datagen.flow(img_array, batch_size=1)
        for i in range(1, num_samples):
            aug_img = next(aug_iter)[0]
            # Normalize to 0-1 for display (original colors preserved)
            aug_img_display = aug_img / 255.0
            aug_img_display = np.clip(aug_img_display, 0, 1)

            axes[i].imshow(aug_img_display)
            axes[i].set_title(f'Augmented {i}', fontsize=10)
            axes[i].axis('off')

        plt.tight_layout()
        assets_dir = Path('../docs/assets')
        assets_dir.mkdir(exist_ok=True)
        plt.savefig(
            assets_dir / f'augmentation_examples_{self.model_variant}.png', dpi=150, bbox_inches='tight')
        print(
            f"\n✓ Augmentation examples saved to {assets_dir / f'augmentation_examples_{self.model_variant}.png'}")
        print("  Note: Original MRI colors are preserved - only geometric transformations applied")
        plt.close()

    def print_preprocessing_summary(self):
        """Print summary of preprocessing configuration."""
        print("\n" + "=" * 80)
        print("DATA PREPARATION SUMMARY")
        print("=" * 80)
        print(f"Model Variant: EfficientNet-{self.model_variant.upper()}")
        print(
            f"Input Size: {self.input_size}×{self.input_size} (square, required by EfficientNet)")
        print(f"Dataset Path: {self.dataset_path.absolute()}")
        print("\nPreprocessing:")
        print("  ✓ Resize with black padding (no cropping)")
        print("  ✓ Black padding is natural for MRI (borders are black, skull is grey)")
        print("  ✓ EfficientNet preprocessing function")
        print("  ✓ RGB format (no conversion needed)")
        print("\nAugmentation Strategy (Geometric Only - Preserves Original Colors):")
        print("  ✓ Zoom out only: 0.95-1.0 (very minimal, prevents cropping)")
        print("  ✓ Translation: ±2% with black padding (very conservative, prevents cropping)")
        print("  ✓ Rotation: ±3° with black padding (very conservative, prevents cropping)")
        print("  ✓ Horizontal flip: Enabled")
        print("  ✓ NO brightness/contrast adjustment - original MRI colors preserved")
        print("\nKey Principles:")
        print("  • All augmentations preserve full image content (no cropping)")
        print("  • Original MRI colors are NEVER changed (only geometric transformations)")
        print("  • Black padding matches natural MRI appearance (black borders, grey skull)")
        print("  • Color changes don't reflect real-world MRI scans")
        print("=" * 80)


def main():
    """Main function to demonstrate data preparation."""
    # Example: Prepare data for EfficientNet B2 (selected candidate)
    prep = DataPreparation(
        dataset_path='../dataset', 
        model_variant='b2',  # Selected: B2, B3, or B4
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Print summary
    prep.print_preprocessing_summary()

    # Step 1: Merge Training and Testing datasets
    print("\n" + "=" * 80)
    print("STEP 1: MERGING DATASETS")
    print("=" * 80)
    prep.merge_datasets(force_merge=False)

    # Step 2: Create data generators with random splitting (no fixed splits)
    print("\n" + "=" * 80)
    print("STEP 2: CREATING DATA GENERATORS (RANDOM SPLITTING)")
    print("=" * 80)
    print("Note: Using random train/val split during training (no fixed splits folder)")
    train_gen, val_gen, test_gen = prep.create_data_generators(
        batch_size=32,
        seed=None,  # None for truly random split each time
        validation_split=0.2,  # 20% for validation
        use_merged=True  # Use merged dataset
    )

    print(f"\n✓ Training samples: {train_gen.samples}")
    print(f"✓ Validation samples: {val_gen.samples}")
    print(f"✓ Test samples: {test_gen.samples}")
    print(f"\nClasses: {train_gen.class_indices}")

    # Step 4: Calculate class weights to handle imbalance
    print("\n" + "=" * 80)
    print("STEP 4: CALCULATING CLASS WEIGHTS")
    print("=" * 80)
    class_weights = prep.get_class_weights(train_gen, method='balanced')

    # Visualize augmentation
    print("\n" + "=" * 80)
    print("STEP 5: VISUALIZING AUGMENTATION")
    print("=" * 80)
    prep.visualize_augmentation()

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  • Merged dataset: {prep.merged_data_path}")
    print(f"  • Random train/val split: 80%/20% (random each time)")
    print(f"  • Test set: Using Testing folder if available")
    print(f"  • Class weights calculated to handle imbalance")
    print(f"  • Safe augmentation preserves all image content")
    print("\nNext Steps:")
    print("1. Use train_gen, val_gen, test_gen for model training")
    print("2. Use class_weights in model.fit() to handle class imbalance")
    print("3. Run: python main.py --model b2 --version v1.0 --epochs 50")
    print("4. Each training run will have a different random train/val split")


if __name__ == '__main__':
    main()
