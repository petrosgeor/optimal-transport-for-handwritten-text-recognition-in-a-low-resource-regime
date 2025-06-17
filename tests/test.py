from pathlib import Path
import sys
from types import SimpleNamespace
import torch

# Add project root to path for imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset

if __name__ == "__main__":
    # Define a simple config object
    class SimpleCfg:
        def __init__(self):
            self.k_external_words = 200
            self.n_aligned = 50
            self.word_emb_dim = 512

    # Create HTRDataset instance
    proj_root = Path(__file__).resolve().parents[1]
    gw_folder = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    
    if not gw_folder.exists():
        raise RuntimeError("GW processed dataset not found – generate it first!")
    
    # Create the dataset instance
    dataset = HTRDataset(
        basefolder=str(gw_folder),
        subset="all",
        fixed_size=(64, 256),
        transforms=None,
        config=SimpleCfg()
    )
    dataset.external_word_histogram()
    
    print(f"Dataset created successfully!")
    print(f"Number of samples: {len(dataset)}")
    print(f"Character classes: {len(dataset.character_classes)}")
    print(f"External words: {len(dataset.external_words)}")