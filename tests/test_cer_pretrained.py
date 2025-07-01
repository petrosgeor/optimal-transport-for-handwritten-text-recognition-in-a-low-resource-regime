import numpy as np
import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add repository root for imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset, PretrainingHTRDataset
from alignment.alignment_utilities import calculate_ot_projections

def get_average_histogram(dataset, num_samples=200):
    """Calculates the average grayscale histogram for a subset of a dataset."""
    histograms = []
    
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i in indices:
        img, _, _ = dataset[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        if img.ndim == 3:
            img = img.squeeze(0)
            
        img_int = (img * 255).astype(np.uint8)
        
        hist, _ = np.histogram(img_int.ravel(), bins=256, range=(0, 255))
        histograms.append(hist)
        
    average_histogram = np.mean(histograms, axis=0)
    return average_histogram

def apply_ot_histogram_matching(image_int, source_hist, target_hist):
    """
    Apply histogram matching using Optimal Transport.
    `image_int` is a numpy array with integer values in [0, 255].
    """
    # Normalize histograms to create probability distributions
    pa = source_hist / (source_hist.sum() + 1e-9)
    pb = target_hist / (target_hist.sum() + 1e-9)
    
    # The "features" are the pixel intensity values themselves
    pixel_values = np.arange(256).reshape(-1, 1).astype(np.float64)
    
    # Get the OT-based mapping
    mapping, _ = calculate_ot_projections(pa, pixel_values, pb, pixel_values, reg=1e-2)
    
    mapping = np.clip(mapping, 0, 255).astype(np.uint8).squeeze()
    
    # Apply the mapping as a lookup table
    transformed_image = mapping[image_int]
    
    return transformed_image

def main():
    """
    Main function to test histogram matching with Optimal Transport.
    """
    # 1. Load the real dataset to compute the average histogram
    real_dataset_path = Path("htr_base/data/GW/processed_words")
    real_dataset = HTRDataset(basefolder=str(real_dataset_path), subset="train_val", fixed_size=(64, 256))
    
    print("Calculating average histogram of the real dataset...")
    target_histogram = get_average_histogram(real_dataset, num_samples=200)

    # 2. Load the synthetic dataset
    synthetic_dataset_config = {
        "list_file": "/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px/imlist.txt",
        "base_path": "/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px",
        "n_random": 1000,
        "fixed_size": [64, 256],
        "preload_images": True,
        "random_seed": 0
    }
    
    try:
        synthetic_dataset = PretrainingHTRDataset(
            list_file=synthetic_dataset_config["list_file"],
            base_path=synthetic_dataset_config["base_path"],
            n_random=synthetic_dataset_config["n_random"],
            fixed_size=tuple(synthetic_dataset_config["fixed_size"]),
            preload_images=synthetic_dataset_config["preload_images"],
            random_seed=synthetic_dataset_config["random_seed"]
        )
    except FileNotFoundError:
        print("Synthetic dataset not found. Skipping histogram matching test.")
        return

    # 3. Perform and visualize histogram matching on a few examples
    num_examples = 5
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 20))
    fig.suptitle("OT Histogram Matching: Synthetic vs. Matched vs. Real Reference", fontsize=16)

    # Get a reference image for visual comparison
    reference_image_tensor, _, _ = real_dataset[np.random.randint(len(real_dataset))]
    reference_image = reference_image_tensor.squeeze().cpu().numpy()

    for i in range(num_examples):
        # Get a synthetic image
        synthetic_image_tensor, _ = synthetic_dataset[i]
        synthetic_image_np = synthetic_image_tensor.squeeze().cpu().numpy()
        synthetic_image_int = (synthetic_image_np * 255).astype(np.uint8)

        # Calculate its histogram
        source_histogram, _ = np.histogram(synthetic_image_int.ravel(), bins=256, range=(0, 255))
        
        # Apply OT matching
        matched_image = apply_ot_histogram_matching(synthetic_image_int, source_histogram, target_histogram)

        # Plotting
        axes[i, 0].imshow(synthetic_image_np, cmap='gray')
        axes[i, 0].set_title("Original Synthetic")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(matched_image, cmap='gray')
        axes[i, 1].set_title("OT Matched")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(reference_image, cmap='gray')
        axes[i, 2].set_title("Reference Real")
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig_path = "tests/figures/ot_histogram_matching_comparison.png"
    Path(fig_path).parent.mkdir(exist_ok=True)
    plt.savefig(fig_path)
    print(f"Saved OT histogram matching comparison to {fig_path}")

if __name__ == "__main__":
    main()