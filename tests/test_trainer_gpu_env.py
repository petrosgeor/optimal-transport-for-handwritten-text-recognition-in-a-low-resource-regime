import sys
import os
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_cuda_visible_devices_set(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "99")
    if "alignment.alignment_trainer" in sys.modules:
        del sys.modules["alignment.alignment_trainer"]
    trainer = importlib.import_module("alignment.alignment_trainer")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(trainer.HP["gpu_id"])
