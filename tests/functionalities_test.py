"""Minimal test suite for htr_new."""

from pathlib import Path
import sys
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.trainer import _shuffle_batch
from htr_base.models import HTRNet
from alignment.trainer import DOMAIN_ADV_ENABLE, refine_visual_backbone
import inspect


def test_trainer_config_has_no_prior_weight():
    """Verify trainer configuration no longer includes ``prior_weight``."""
    cfg = OmegaConf.load("alignment/alignment_configs/trainer_config.yaml")
    assert "prior_weight" not in cfg


def test_shuffle_batch():
    """_shuffle_batch keeps pairs together and randomises order."""
    torch.manual_seed(0)
    imgs = torch.arange(6).view(3, 2)
    words = ["a", "b", "c"]
    shuffled_imgs, shuffled_words, extra = _shuffle_batch(imgs.clone(), list(words))
    assert extra is None

    expected_pairs = [
        ((4, 5), "c"),
        ((0, 1), "a"),
        ((2, 3), "b"),
    ]
    observed_pairs = [
        (tuple(shuffled_imgs[i].tolist()), shuffled_words[i])
        for i in range(3)
    ]
    assert observed_pairs == expected_pairs
    assert not torch.equal(imgs, shuffled_imgs)


def test_shuffle_batch_with_extra():
    """_shuffle_batch shuffles an additional tensor consistently."""
    torch.manual_seed(0)
    imgs = torch.arange(6).view(3, 2)
    words = ["a", "b", "c"]
    extra = torch.tensor([10, 11, 12])
    s_imgs, s_words, s_extra = _shuffle_batch(imgs.clone(), list(words), extra.clone())

    assert [w for w in s_words] == ["c", "a", "b"]
    assert s_extra.tolist() == [12, 10, 11]
    assert not torch.equal(extra, s_extra)


def test_grl_lambda_linear():
    """_grl_lambda returns linear ramp when mode='linear'."""
    from alignment.trainer import _grl_lambda

    assert _grl_lambda(0, 10, mode="linear") == 0.0
    assert abs(_grl_lambda(5, 10, mode="linear") - 0.5) < 1e-6
    assert _grl_lambda(10, 10, mode="linear") == 1.0


def test_domain_head():
    """DomainHead outputs a (B,) logit tensor."""
    from types import SimpleNamespace

    arch = SimpleNamespace(
        cnn_cfg=[[2, 64], 'M', [3, 128], 'M', [2, 256]],
        head_type='both', rnn_type='gru', rnn_layers=3,
        rnn_hidden_size=256, flattening='maxpool',
        feat_dim=512, feat_pool='avg', stn=False,
    )
    net = HTRNet(arch, nclasses=38)
    x = torch.rand(4, 1, 128, 512)
    main, aux, feats = net(x, return_feats=True)
    dom = net.domain_head(feats, 1.0)
    assert dom.shape == (4,)


def test_domain_adv_flag_loaded():
    """DOMAIN_ADV_ENABLE should match YAML setting."""
    cfg = OmegaConf.load("alignment/alignment_configs/trainer_config.yaml")
    assert DOMAIN_ADV_ENABLE == cfg.domain_adv_enable


def test_refine_visual_backbone_defaults():
    """Function defaults use the parsed domain-adversarial flag."""
    sig = inspect.signature(refine_visual_backbone)
    assert sig.parameters["enable_domain_adv"].default == DOMAIN_ADV_ENABLE

