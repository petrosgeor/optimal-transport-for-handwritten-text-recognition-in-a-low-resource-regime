import sys, random
from pathlib import Path


def test_concat_prob_fixed_width_and_transcription():

    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    from htr_base.utils.htr_dataset import HTRDataset

    ds_path = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    ds = HTRDataset(str(ds_path), subset="train", fixed_size=(32, 64),
                    transforms=None, concat_prob=1.0)
    random.seed(0)
    random.random()  # value used internally
    idx2 = random.randint(0, len(ds.data) - 1)
    random.seed(0)
    img, transcr, _ = ds[0]
    print("single", img.shape, transcr)
    assert img.shape[-2:] == ds.fixed_size
    expected = f" {ds.data[0][1].strip()}   {ds.data[idx2][1].strip()} "
    assert transcr == expected

    # Verify DataLoader integration
    from torch.utils.data import DataLoader
    random.seed(0)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch_imgs, batch_transcrs, _ = next(iter(loader))
    print("batch", batch_imgs.shape, batch_transcrs)
    assert tuple(batch_imgs.shape[1:]) == (1, *ds.fixed_size)
    for img_t in batch_imgs:
        assert img_t.shape[-2:] == ds.fixed_size
    assert batch_transcrs[0] == expected

