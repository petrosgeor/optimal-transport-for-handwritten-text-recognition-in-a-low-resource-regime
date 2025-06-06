import sys
from pathlib import Path


class DummyCfg:
    def __init__(self, k_external_words: int = 30):
        self.k_external_words = k_external_words
        self.n_aligned = 0
        self.word_emb_dim = 32


def test_external_vocab_characters() -> None:
    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    from htr_base.utils.htr_dataset import HTRDataset

    dataset_path = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    ds = HTRDataset(str(dataset_path), subset="train", fixed_size=(64, 256),
                    transforms=None, config=DummyCfg())

    for word in ds.external_words:
        for ch in word:
            assert ch in ds.character_classes

