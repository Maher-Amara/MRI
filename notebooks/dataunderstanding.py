"""
Data Understanding Script for Brain Tumor Detection Project
Phase 2 of CRISP-DM Methodology

This script analyzes the MRI brain tumor dataset and generates a comprehensive report
including statistics, visualizations, and data quality assessments.

Requirements:
    pip install numpy pillow matplotlib seaborn

Usage:
    python dataunderstanding.py
    OR
    from dataunderstanding import DataUnderstanding
    analyzer = DataUnderstanding(dataset_path='../dataset')
    analyzer.run_full_analysis()
"""

import os
import sys
from pathlib import Path
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow (PIL) not installed. Install with: pip install pillow")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: Matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)

try:
    import seaborn as sns
except ImportError:
    print("WARNING: Seaborn not installed. Some visualizations may not work.")
    print("Install with: pip install seaborn")
    sns = None

from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations (if seaborn is available)
if sns is not None:
    try:
        sns.set_style("whitegrid")
    except:
        pass
plt.rcParams['figure.figsize'] = (12, 8)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class DataUnderstanding:
    """Class to analyze and understand the brain tumor MRI dataset."""

    def __init__(self, dataset_path='../dataset'):
        """
        Initialize the DataUnderstanding class.

        Args:
            dataset_path (str): Path to the dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.training_path = self.dataset_path / 'Training'
        self.testing_path = self.dataset_path / 'Testing'

        # Data storage
        self.stats = {
            'training': defaultdict(int),
            'testing': defaultdict(int),
            'image_properties': {
                'dimensions': [],
                'file_sizes': [],
                'formats': defaultdict(int),
                'channels': defaultdict(int)
            },
            'corrupted_images': [],
            'class_distribution': defaultdict(int)
        }

    def check_dataset_structure(self):
        """Check if dataset directory structure is correct."""
        print("=" * 80)
        print("DATASET STRUCTURE CHECK")
        print("=" * 80)

        if not self.dataset_path.exists():
            print(
                f"❌ ERROR: Dataset path '{self.dataset_path}' does not exist!")
            return False

        print(f"✓ Dataset path found: {self.dataset_path.absolute()}")

        # Check Training directory
        if self.training_path.exists():
            print(f"✓ Training directory found: {self.training_path}")
        else:
            print(f"❌ Training directory not found: {self.training_path}")
            return False

        # Check Testing directory
        if self.testing_path.exists():
            print(f"✓ Testing directory found: {self.testing_path}")
        else:
            print(f"❌ Testing directory not found: {self.testing_path}")
            return False

        # Check for class directories
        print("\nChecking class directories...")
        expected_classes = ['glioma_tumor',
                            'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

        for split in ['Training', 'Testing']:
            split_path = self.dataset_path / split
            if split_path.exists():
                print(f"\n{split} classes:")
                for class_name in expected_classes:
                    class_path = split_path / class_name
                    if class_path.exists():
                        count = len(list(class_path.glob('*.jpg'))) + len(
                            list(class_path.glob('*.jpeg'))) + len(list(class_path.glob('*.png')))
                        print(f"  ✓ {class_name}: {count} images")
                    else:
                        print(f"  ❌ {class_name}: Not found")

        return True

    def count_images(self):
        """Count images in each class for both training and testing sets."""
        print("\n" + "=" * 80)
        print("IMAGE COUNT ANALYSIS")
        print("=" * 80)

        splits = {
            'Training': self.training_path,
            'Testing': self.testing_path
        }

        total_images = 0

        for split_name, split_path in splits.items():
            if not split_path.exists():
                continue

            print(f"\n{split_name} Set:")
            print("-" * 80)

            # Get all class directories
            class_dirs = [d for d in split_path.iterdir() if d.is_dir()]

            for class_dir in sorted(class_dirs):
                class_name = class_dir.name
                # Count image files
                image_files = list(class_dir.glob(
                    '*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
                count = len(image_files)

                self.stats[split_name.lower()][class_name] = count
                self.stats['class_distribution'][class_name] += count
                total_images += count

                print(f"  {class_name:25s}: {count:4d} images")

            split_total = sum(self.stats[split_name.lower()].values())
            print(f"  {'TOTAL':25s}: {split_total:4d} images")

        print(f"\n{'=' * 80}")
        print(f"GRAND TOTAL: {total_images} images")
        print("=" * 80)

        return total_images

    def analyze_image_properties(self, sample_size=None):
        """
        Analyze properties of images in the dataset.

        Args:
            sample_size (int): Number of images to sample for analysis. If None, analyzes all.
        """
        print("\n" + "=" * 80)
        print("IMAGE PROPERTIES ANALYSIS")
        print("=" * 80)

        splits = {
            'Training': self.training_path,
            'Testing': self.testing_path
        }

        all_images = []
        corrupted_count = 0

        # Collect all image paths
        for split_path in splits.values():
            if not split_path.exists():
                continue
            for class_dir in split_path.iterdir():
                if class_dir.is_dir():
                    image_files = list(class_dir.glob(
                        '*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
                    all_images.extend(image_files)

        # Sample if needed
        if sample_size and sample_size < len(all_images):
            import random
            random.seed(42)
            all_images = random.sample(all_images, sample_size)
            print(f"Sampling {sample_size} images for analysis...")
        else:
            print(f"Analyzing all {len(all_images)} images...")

        print("\nAnalyzing image properties...")

        for idx, img_path in enumerate(all_images):
            if (idx + 1) % 100 == 0:
                print(
                    f"  Processed {idx + 1}/{len(all_images)} images...", end='\r')

            try:
                # Get file size
                file_size = img_path.stat().st_size / (1024 * 1024)  # Size in MB
                self.stats['image_properties']['file_sizes'].append(file_size)

                # Get image format
                ext = img_path.suffix.lower()
                self.stats['image_properties']['formats'][ext] += 1

                # Open and analyze image
                with Image.open(img_path) as img:
                    width, height = img.size
                    self.stats['image_properties']['dimensions'].append(
                        (width, height))

                    # Check channels
                    if img.mode == 'RGB':
                        channels = 3
                    elif img.mode == 'L':
                        channels = 1
                    elif img.mode == 'RGBA':
                        channels = 4
                    else:
                        channels = len(img.getbands())

                    self.stats['image_properties']['channels'][channels] += 1

            except Exception as e:
                corrupted_count += 1
                self.stats['corrupted_images'].append({
                    'path': str(img_path),
                    'error': str(e)
                })

        print(f"\n✓ Analysis complete! Processed {len(all_images)} images")
        if corrupted_count > 0:
            print(f"⚠ Found {corrupted_count} corrupted/unreadable images")

        return len(all_images)

    def print_statistics(self):
        """Print comprehensive statistics about the dataset."""
        print("\n" + "=" * 80)
        print("DETAILED STATISTICS")
        print("=" * 80)

        # Image counts
        print("\n1. IMAGE COUNTS BY CLASS AND SPLIT")
        print("-" * 80)
        print(f"{'Class':<25} {'Training':<12} {'Testing':<12} {'Total':<12}")
        print("-" * 80)

        all_classes = set()
        all_classes.update(self.stats['training'].keys())
        all_classes.update(self.stats['testing'].keys())

        for class_name in sorted(all_classes):
            train_count = self.stats['training'].get(class_name, 0)
            test_count = self.stats['testing'].get(class_name, 0)
            total = train_count + test_count
            print(f"{class_name:<25} {train_count:<12} {test_count:<12} {total:<12}")

        train_total = sum(self.stats['training'].values())
        test_total = sum(self.stats['testing'].values())
        grand_total = train_total + test_total
        print("-" * 80)
        print(f"{'TOTAL':<25} {train_total:<12} {test_total:<12} {grand_total:<12}")

        # Class distribution percentages
        print("\n2. CLASS DISTRIBUTION (PERCENTAGES)")
        print("-" * 80)
        total = sum(self.stats['class_distribution'].values())
        for class_name in sorted(self.stats['class_distribution'].keys()):
            count = self.stats['class_distribution'][class_name]
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{class_name:<25}: {count:4d} ({percentage:5.2f}%)")

        # Image dimensions
        if self.stats['image_properties']['dimensions']:
            print("\n3. IMAGE DIMENSIONS")
            print("-" * 80)
            dimensions = np.array(self.stats['image_properties']['dimensions'])
            widths = dimensions[:, 0]
            heights = dimensions[:, 1]

            print(f"Width  - Min: {int(widths.min())}, Max: {int(widths.max())}, "
                  f"Mean: {widths.mean():.1f}, Std: {widths.std():.1f}")
            print(f"Height - Min: {int(heights.min())}, Max: {int(heights.max())}, "
                  f"Mean: {heights.mean():.1f}, Std: {heights.std():.1f}")

            # Most common dimensions
            unique_dims, counts = np.unique(
                dimensions, axis=0, return_counts=True)
            top_dims = unique_dims[np.argsort(counts)[-5:]][::-1]
            top_counts = counts[np.argsort(counts)[-5:]][::-1]

            print("\nMost common dimensions:")
            for dim, count in zip(top_dims, top_counts):
                print(f"  {int(dim[0])}x{int(dim[1])}: {count} images")

        # File sizes
        if self.stats['image_properties']['file_sizes']:
            print("\n4. FILE SIZES")
            print("-" * 80)
            file_sizes = np.array(self.stats['image_properties']['file_sizes'])
            print(f"Min: {file_sizes.min():.3f} MB")
            print(f"Max: {file_sizes.max():.3f} MB")
            print(f"Mean: {file_sizes.mean():.3f} MB")
            print(f"Median: {np.median(file_sizes):.3f} MB")
            print(f"Std: {file_sizes.std():.3f} MB")
            print(
                f"Total size: {file_sizes.sum():.2f} MB ({file_sizes.sum()/1024:.2f} GB)")

        # Formats
        if self.stats['image_properties']['formats']:
            print("\n5. FILE FORMATS")
            print("-" * 80)
            for fmt, count in sorted(self.stats['image_properties']['formats'].items(),
                                     key=lambda x: x[1], reverse=True):
                print(f"{fmt:<10}: {count:4d} images")

        # Channels
        if self.stats['image_properties']['channels']:
            print("\n6. COLOR CHANNELS")
            print("-" * 80)
            for channels, count in sorted(self.stats['image_properties']['channels'].items()):
                channel_type = {1: 'Grayscale', 3: 'RGB', 4: 'RGBA'}.get(
                    channels, f'{channels} channels')
                print(f"{channel_type:<15}: {count:4d} images")

        # Corrupted images
        if self.stats['corrupted_images']:
            print("\n7. CORRUPTED IMAGES")
            print("-" * 80)
            print(
                f"Found {len(self.stats['corrupted_images'])} corrupted/unreadable images:")
            for corrupted in self.stats['corrupted_images'][:10]:  # Show first 10
                print(f"  - {corrupted['path']}")
                print(f"    Error: {corrupted['error']}")
            if len(self.stats['corrupted_images']) > 10:
                print(
                    f"  ... and {len(self.stats['corrupted_images']) - 10} more")

        # Data quality assessment
        print("\n8. DATA QUALITY ASSESSMENT")
        print("-" * 80)

        # Check class imbalance
        class_counts = list(self.stats['class_distribution'].values())
        if class_counts:
            max_count = max(class_counts)
            min_count = min(class_counts)
            imbalance_ratio = max_count / \
                min_count if min_count > 0 else float('inf')

            if imbalance_ratio > 2.0:
                print(
                    f"⚠ Class imbalance detected! Ratio: {imbalance_ratio:.2f}")
                print(
                    "  Recommendation: Consider using class weights or data augmentation")
            else:
                print("✓ Classes are relatively balanced")

        # Check for missing classes
        expected_classes = ['glioma_tumor',
                            'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        missing_classes = [
            c for c in expected_classes if c not in self.stats['class_distribution']]
        if missing_classes:
            print(f"⚠ Missing classes: {', '.join(missing_classes)}")
        else:
            print("✓ All expected classes are present")

        # Check train/test split
        train_total = sum(self.stats['training'].values())
        test_total = sum(self.stats['testing'].values())
        if train_total > 0 and test_total > 0:
            test_ratio = test_total / (train_total + test_total)
            print(f"✓ Train/Test split: {train_total}/{test_total} "
                  f"({(1-test_ratio)*100:.1f}%/{test_ratio*100:.1f}%)")
            if test_ratio < 0.1:
                print("  ⚠ Warning: Test set is very small (<10%)")
            elif test_ratio > 0.3:
                print("  ⚠ Warning: Test set is large (>30%)")

    def visualize_sample_images(self, num_samples=4):
        """
        Visualize sample images from each class.

        Args:
            num_samples (int): Number of sample images per class to display
        """
        print("\n" + "=" * 80)
        print("SAMPLE IMAGES VISUALIZATION")
        print("=" * 80)

        splits = {
            'Training': self.training_path,
            'Testing': self.testing_path
        }

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Sample Images from Each Class',
                     fontsize=16, fontweight='bold')

        class_names = ['glioma_tumor', 'meningioma_tumor',
                       'no_tumor', 'pituitary_tumor']

        for col, class_name in enumerate(class_names):
            # Training sample
            train_path = self.training_path / class_name
            if train_path.exists():
                train_images = list(train_path.glob(
                    '*.jpg')) + list(train_path.glob('*.jpeg')) + list(train_path.glob('*.png'))
                if train_images:
                    img = Image.open(train_images[0])
                    axes[0, col].imshow(
                        img, cmap='gray' if img.mode == 'L' else None)
                    axes[0, col].set_title(
                        f'Training: {class_name}', fontsize=10)
                    axes[0, col].axis('off')

            # Testing sample
            test_path = self.testing_path / class_name
            if test_path.exists():
                test_images = list(test_path.glob(
                    '*.jpg')) + list(test_path.glob('*.jpeg')) + list(test_path.glob('*.png'))
                if test_images:
                    img = Image.open(test_images[0])
                    axes[1, col].imshow(
                        img, cmap='gray' if img.mode == 'L' else None)
                    axes[1, col].set_title(
                        f'Testing: {class_name}', fontsize=10)
                    axes[1, col].axis('off')

        plt.tight_layout()
        assets_dir = Path('../docs/assets')
        assets_dir.mkdir(exist_ok=True)
        plt.savefig(assets_dir / 'sample_images.png', dpi=150, bbox_inches='tight')
        print(f"✓ Sample images saved to {assets_dir / 'sample_images.png'}")
        plt.close()

    def visualize_class_distribution(self):
        """Create visualizations for class distribution."""
        print("\n" + "=" * 80)
        print("CLASS DISTRIBUTION VISUALIZATIONS")
        print("=" * 80)

        # Prepare data
        classes = sorted(self.stats['class_distribution'].keys())
        train_counts = [self.stats['training'].get(c, 0) for c in classes]
        test_counts = [self.stats['testing'].get(c, 0) for c in classes]

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar plot
        x = np.arange(len(classes))
        width = 0.35

        axes[0].bar(x - width/2, train_counts, width,
                    label='Training', color='#3498db')
        axes[0].bar(x + width/2, test_counts, width,
                    label='Testing', color='#e74c3c')
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Number of Images', fontsize=12)
        axes[0].set_title('Image Count by Class and Split',
                          fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([c.replace('_', ' ').title()
                                for c in classes], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Pie chart for total distribution
        total_counts = [train_counts[i] + test_counts[i]
                        for i in range(len(classes))]
        axes[1].pie(total_counts, labels=[c.replace('_', ' ').title() for c in classes],
                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        axes[1].set_title('Overall Class Distribution',
                          fontsize=14, fontweight='bold')

        plt.tight_layout()
        assets_dir = Path('../docs/assets')
        assets_dir.mkdir(exist_ok=True)
        plt.savefig(assets_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
        print(f"✓ Class distribution plots saved to {assets_dir / 'class_distribution.png'}")
        plt.close()
    
    def visualize_image_dimensions(self):
        """Create visualization for image dimensions distribution."""
        if not self.stats['image_properties']['dimensions']:
            print("⚠ No dimension data available for visualization")
            return
        
        print("\n" + "=" * 80)
        print("IMAGE DIMENSIONS VISUALIZATION")
        print("=" * 80)
        
        dimensions = np.array(self.stats['image_properties']['dimensions'])
        widths = dimensions[:, 0]
        heights = dimensions[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Image Dimensions Analysis', fontsize=16, fontweight='bold')
        
        # Width distribution
        axes[0, 0].hist(widths, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(widths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {widths.mean():.1f}')
        axes[0, 0].set_xlabel('Width (pixels)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Width Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Height distribution
        axes[0, 1].hist(heights, bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(heights.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {heights.mean():.1f}')
        axes[0, 1].set_xlabel('Height (pixels)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Height Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Scatter plot of dimensions
        axes[1, 0].scatter(widths, heights, alpha=0.3, s=10, color='#2ecc71')
        axes[1, 0].axline([0, 0], [1, 1], color='red', linestyle='--', linewidth=2, label='Square (1:1)')
        axes[1, 0].set_xlabel('Width (pixels)', fontsize=12)
        axes[1, 0].set_ylabel('Height (pixels)', fontsize=12)
        axes[1, 0].set_title('Width vs Height Scatter', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Most common dimensions
        unique_dims, counts = np.unique(dimensions, axis=0, return_counts=True)
        top_10_idx = np.argsort(counts)[-10:][::-1]
        top_dims = unique_dims[top_10_idx]
        top_counts = counts[top_10_idx]
        
        dim_labels = [f"{int(w)}×{int(h)}" for w, h in top_dims]
        axes[1, 1].barh(range(len(dim_labels)), top_counts, color='#9b59b6')
        axes[1, 1].set_yticks(range(len(dim_labels)))
        axes[1, 1].set_yticklabels(dim_labels)
        axes[1, 1].set_xlabel('Number of Images', fontsize=12)
        axes[1, 1].set_title('Top 10 Most Common Dimensions', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        assets_dir = Path('../docs/assets')
        assets_dir.mkdir(exist_ok=True)
        plt.savefig(assets_dir / 'image_dimensions.png', dpi=150, bbox_inches='tight')
        print(f"✓ Image dimensions visualization saved to {assets_dir / 'image_dimensions.png'}")
        plt.close()
    
    def visualize_file_properties(self):
        """Create visualizations for file formats, sizes, and channels."""
        print("\n" + "=" * 80)
        print("FILE PROPERTIES VISUALIZATION")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('File Properties Analysis', fontsize=16, fontweight='bold')
        
        # File formats
        if self.stats['image_properties']['formats']:
            formats = list(self.stats['image_properties']['formats'].keys())
            counts = list(self.stats['image_properties']['formats'].values())
            axes[0, 0].bar(formats, counts, color='#3498db', edgecolor='black')
            axes[0, 0].set_xlabel('File Format', fontsize=12)
            axes[0, 0].set_ylabel('Number of Images', fontsize=12)
            axes[0, 0].set_title('File Format Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].grid(axis='y', alpha=0.3)
            for i, (fmt, count) in enumerate(zip(formats, counts)):
                axes[0, 0].text(i, count, str(count), ha='center', va='bottom')
        else:
            axes[0, 0].text(0.5, 0.5, 'No format data', ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # File sizes
        if self.stats['image_properties']['file_sizes']:
            file_sizes = np.array(self.stats['image_properties']['file_sizes'])
            axes[0, 1].hist(file_sizes, bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(file_sizes.mean(), color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {file_sizes.mean():.3f} MB')
            axes[0, 1].set_xlabel('File Size (MB)', fontsize=12)
            axes[0, 1].set_ylabel('Frequency', fontsize=12)
            axes[0, 1].set_title('File Size Distribution', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(axis='y', alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No file size data', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Color channels
        if self.stats['image_properties']['channels']:
            channels = list(self.stats['image_properties']['channels'].keys())
            counts = list(self.stats['image_properties']['channels'].values())
            channel_labels = {1: 'Grayscale', 3: 'RGB', 4: 'RGBA'}
            labels = [channel_labels.get(c, f'{c} channels') for c in channels]
            axes[1, 0].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
                          colors=['#2ecc71', '#3498db', '#9b59b6', '#f39c12'])
            axes[1, 0].set_title('Color Channel Distribution', fontsize=14, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No channel data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Train/Test split visualization
        train_total = sum(self.stats['training'].values())
        test_total = sum(self.stats['testing'].values())
        if train_total > 0 or test_total > 0:
            axes[1, 1].pie([train_total, test_total], labels=['Training', 'Testing'], 
                          autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
            axes[1, 1].set_title('Train/Test Split', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No split data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        assets_dir = Path('../docs/assets')
        assets_dir.mkdir(exist_ok=True)
        plt.savefig(assets_dir / 'file_properties.png', dpi=150, bbox_inches='tight')
        print(f"✓ File properties visualization saved to {assets_dir / 'file_properties.png'}")
        plt.close()

    def generate_report(self):
        """Generate a comprehensive text report and save it to a file."""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 80)

        report_path = Path('../docs/data_understanding_report.txt')
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BRAIN TUMOR DETECTION - DATA UNDERSTANDING REPORT\n")
            f.write("Phase 2 of CRISP-DM Methodology\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Dataset Path: {self.dataset_path.absolute()}\n")
            f.write(f"Training Path: {self.training_path.absolute()}\n")
            f.write(f"Testing Path: {self.testing_path.absolute()}\n\n")

            f.write("2. IMAGE COUNTS\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Class':<25} {'Training':<12} {'Testing':<12} {'Total':<12}\n")
            f.write("-" * 80 + "\n")

            all_classes = set()
            all_classes.update(self.stats['training'].keys())
            all_classes.update(self.stats['testing'].keys())

            for class_name in sorted(all_classes):
                train_count = self.stats['training'].get(class_name, 0)
                test_count = self.stats['testing'].get(class_name, 0)
                total = train_count + test_count
                f.write(
                    f"{class_name:<25} {train_count:<12} {test_count:<12} {total:<12}\n")

            train_total = sum(self.stats['training'].values())
            test_total = sum(self.stats['testing'].values())
            grand_total = train_total + test_total
            f.write("-" * 80 + "\n")
            f.write(
                f"{'TOTAL':<25} {train_total:<12} {test_total:<12} {grand_total:<12}\n\n")

            f.write("3. CLASS DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            total = sum(self.stats['class_distribution'].values())
            for class_name in sorted(self.stats['class_distribution'].keys()):
                count = self.stats['class_distribution'][class_name]
                percentage = (count / total) * 100 if total > 0 else 0
                f.write(f"{class_name:<25}: {count:4d} ({percentage:5.2f}%)\n")
            f.write("\n")

            if self.stats['image_properties']['dimensions']:
                f.write("4. IMAGE PROPERTIES\n")
                f.write("-" * 80 + "\n")
                dimensions = np.array(
                    self.stats['image_properties']['dimensions'])
                widths = dimensions[:, 0]
                heights = dimensions[:, 1]

                f.write(f"Width  - Min: {int(widths.min())}, Max: {int(widths.max())}, "
                        f"Mean: {widths.mean():.1f}, Std: {widths.std():.1f}\n")
                f.write(f"Height - Min: {int(heights.min())}, Max: {int(heights.max())}, "
                        f"Mean: {heights.mean():.1f}, Std: {heights.std():.1f}\n\n")

            if self.stats['image_properties']['file_sizes']:
                f.write("5. FILE SIZES\n")
                f.write("-" * 80 + "\n")
                file_sizes = np.array(
                    self.stats['image_properties']['file_sizes'])
                f.write(f"Min: {file_sizes.min():.3f} MB\n")
                f.write(f"Max: {file_sizes.max():.3f} MB\n")
                f.write(f"Mean: {file_sizes.mean():.3f} MB\n")
                f.write(f"Median: {np.median(file_sizes):.3f} MB\n")
                f.write(
                    f"Total: {file_sizes.sum():.2f} MB ({file_sizes.sum()/1024:.2f} GB)\n\n")

            f.write("6. DATA QUALITY ASSESSMENT\n")
            f.write("-" * 80 + "\n")
            if self.stats['corrupted_images']:
                f.write(
                    f"Corrupted Images: {len(self.stats['corrupted_images'])}\n")
            else:
                f.write("Corrupted Images: 0\n")

            # Class imbalance check
            class_counts = list(self.stats['class_distribution'].values())
            if class_counts:
                max_count = max(class_counts)
                min_count = min(class_counts)
                imbalance_ratio = max_count / \
                    min_count if min_count > 0 else float('inf')
                f.write(f"Class Imbalance Ratio: {imbalance_ratio:.2f}\n")

            f.write("\n7. RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("- Standardize image dimensions during preprocessing\n")
            f.write(
                "- Consider data augmentation to handle class imbalance if present\n")
            f.write(
                "- Split training data into train/validation sets (80/20 or 70/30)\n")
            f.write("- Normalize pixel values to [0, 1] or [-1, 1] range\n")
            if self.stats['corrupted_images']:
                f.write("- Remove or fix corrupted images before training\n")

        print(f"✓ Comprehensive report saved to {report_path}")

    def run_full_analysis(self, sample_size=None):
        """
        Run the complete data understanding analysis.

        Args:
            sample_size (int): Number of images to sample for property analysis. 
                             If None, analyzes all images.
        """
        print("\n" + "=" * 80)
        print("BRAIN TUMOR DETECTION - DATA UNDERSTANDING ANALYSIS")
        print("Phase 2: CRISP-DM Methodology")
        print("=" * 80)

        # Step 1: Check dataset structure
        if not self.check_dataset_structure():
            print("\n❌ Dataset structure check failed. Please verify the dataset path.")
            return

        # Step 2: Count images
        total_images = self.count_images()

        # Step 3: Analyze image properties
        if total_images > 0:
            self.analyze_image_properties(sample_size=sample_size)

        # Step 4: Print statistics
        self.print_statistics()

        # Step 5: Visualizations
        if total_images > 0:
            try:
                self.visualize_sample_images()
                self.visualize_class_distribution()
                self.visualize_image_dimensions()
                self.visualize_file_properties()
            except Exception as e:
                print(f"\n⚠ Warning: Could not generate visualizations: {e}")
                import traceback
                traceback.print_exc()

        # Step 6: Generate report
        self.generate_report()

        print("\n" + "=" * 80)
        print("DATA UNDERSTANDING ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated Files:")
        print("1. Report: docs/data_understanding_report.txt")
        print("2. Visualizations: docs/assets/")
        print("   - sample_images.png")
        print("   - class_distribution.png")
        print("   - image_dimensions.png")
        print("   - file_properties.png")
        print("\nNext Steps:")
        print("1. Review the generated report and visualizations")
        print("2. Proceed to Phase 3: Data Preparation")


def main():
    """Main function to run the data understanding analysis."""
    # Initialize the analyzer
    analyzer = DataUnderstanding(dataset_path='../dataset')

    # Run full analysis
    # For large datasets, you can specify a sample_size to speed up property analysis
    # analyzer.run_full_analysis(sample_size=1000)  # Analyze 1000 random images
    analyzer.run_full_analysis()  # Analyze all images


if __name__ == '__main__':
    main()
