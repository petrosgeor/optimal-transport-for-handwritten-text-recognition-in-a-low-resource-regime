import sys, random
from pathlib import Path


def test_concat_prob_doubles_width_and_transcription():
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
    assert img.shape[-1] == 2 * ds.fixed_size[1]
    expected = f" {ds.data[0][1].strip()}   {ds.data[idx2][1].strip()} "
    assert transcr == expected
