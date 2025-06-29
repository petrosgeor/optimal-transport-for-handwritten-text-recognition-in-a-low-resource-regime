import importlib
import sys
import logging
import torch

def test_beam_search_import_no_kenlm_warning(caplog):
    module = 'alignment.ctc_utils'
    if module in sys.modules:
        del sys.modules[module]
    caplog.set_level(logging.WARNING)
    with caplog.at_level(logging.WARNING, logger='pyctcdecode.decoder'):
        ctcu = importlib.import_module(module)
    assert not any('kenlm python bindings' in r.message for r in caplog.records)
    logits = torch.zeros(2, 1, 3)
    preds = ctcu.beam_search_ctc_decode(logits, {1: 'a', 2: 'b'})
    assert preds == [""]
