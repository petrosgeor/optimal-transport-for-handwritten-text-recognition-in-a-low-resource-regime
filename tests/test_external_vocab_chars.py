from __future__ import annotations
import sys
from pathlib import Path


class DummyCfg:
    def __init__(self, k_external_words: int = 200):
        self.k_external_words = k_external_words
        self.n_aligned = 0
        self.word_emb_dim = 512


def main() -> None:
    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    from htr_base.utils.htr_dataset import HTRDataset

    dataset_path = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset path not found: {dataset_path}")

    ds = HTRDataset(str(dataset_path), subset="train", fixed_size=(64, 256),
                    transforms=None, config=DummyCfg())

    ext_chars = sorted({ch for w in ds.external_words for ch in w})
    ds_chars = sorted(set(ds.character_classes))

    print(f"External vocab chars ({len(ext_chars)}): {ext_chars}")
    print(f"Dataset chars ({len(ds_chars)}): {ds_chars}")

    diff_ext = sorted(set(ext_chars) - set(ds_chars))
    diff_ds = sorted(set(ds_chars) - set(ext_chars))

    print(f"Chars only in external words: {diff_ext}")
    print(f"Chars absent from external words: {diff_ds}")


if __name__ == "__main__":
    main()
