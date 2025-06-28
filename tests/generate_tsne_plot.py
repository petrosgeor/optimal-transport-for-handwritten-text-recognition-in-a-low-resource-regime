import sys
from pathlib import Path
import torch

# Add project root to path to allow for relative imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
from alignment.plot import plot_pretrained_backbone_tsne

def generate_plot():
    """
    Generates and saves a t-SNE plot of the pretrained backbone embeddings.
    """
    print("Initializing dataset...")
    # The path to the dataset is derived from the project structure
    gw_folder = root / "htr_base" / "data" / "GW" / "processed_words"

    # Instantiate the dataset
    dataset = HTRDataset(
        basefolder=str(gw_folder),
        subset="all",
        fixed_size=(64, 256),
    )

    n_samples = 800
    save_path = str(root / "tests" / "figures" / "pretrained_backbone_tsne.png")

    print(f"Generating t-SNE plot for {n_samples} samples...")
    
    # Generate and save the plot
    plot_pretrained_backbone_tsne(
        dataset=dataset,
        n_samples=n_samples,
        save_path=save_path
    )

    print(f"t-SNE plot saved to {save_path}")

if __name__ == "__main__":
    generate_plot()
