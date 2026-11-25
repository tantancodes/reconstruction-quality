"""python -m pytest experiments/geometry-aware/test_geo_models.py -q"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_models as gm  # noqa: E402


def random_rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.where(np.diag(r) < 0, -1.0, 1.0)
    return torch.tensor(q if np.linalg.det(q) > 0 else -q, dtype=torch.float32)


def test_no_bias_equals_torch_multihead_attention():
    torch.manual_seed(0)
    h = 32
    a = gm.GeoAttention(h, 4, rel_dim=0)
    m = torch.nn.MultiheadAttention(h, 4, batch_first=True)
    with torch.no_grad():
        m.in_proj_weight.copy_(a.qkv.weight)
        m.in_proj_bias.copy_(a.qkv.bias)
        m.out_proj.weight.copy_(a.out.weight)
        m.out_proj.bias.copy_(a.out.bias)
    f = torch.randn(3, 7, h)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[0, 5:] = False
    assert torch.allclose(a(f, mask), m(f, f, f, key_padding_mask=~mask)[0], atol=1e-5)


def test_relations_are_rotation_invariant():
    torch.manual_seed(1)
    P, V = torch.randn(2, 9, 3), torch.nn.functional.normalize(torch.randn(2, 9, 3), dim=-1)
    R = random_rotation(2)
    assert torch.allclose(gm.relations(P @ R.T, V @ R.T), gm.relations(P, V), atol=1e-5)


def test_geo_inv_model_is_rotation_invariant():
    torch.manual_seed(3)
    P = torch.randn(2, 9, 3)
    M = torch.ones(2, 9, dtype=torch.bool)
    model = gm.make_model('geo_inv', 3, 16, 2).eval()
    R = random_rotation(4)
    out = model(gm.invariant_tokens(P), M, gm.relations(P, None, M))
    out_r = model(gm.invariant_tokens(P @ R.T), M, gm.relations(P @ R.T, None, M))
    assert torch.allclose(out, out_r, atol=1e-5)


def test_shuffle_keeps_values_and_padding():
    torch.manual_seed(5)
    M = torch.zeros(2, 5, dtype=torch.bool)
    M[0, :3] = True
    M[1, :] = True
    rel = gm.relations(torch.randn(2, 5, 3), None, M)
    out = gm.shuffle_pairs(rel, M, torch.Generator().manual_seed(0))
    assert torch.allclose(rel[0, :3, :3].flatten().sort().values, out[0, :3, :3].flatten().sort().values)
    assert out[0, 3:].abs().sum() == 0


def test_permutation_invariance():
    torch.manual_seed(6)
    P = torch.randn(1, 8, 3)
    M = torch.ones(1, 8, dtype=torch.bool)
    model = gm.make_model('geo', 3, 16, 2).eval()
    perm = torch.randperm(8)
    a = model(P, M, gm.relations(P, None, M))
    b = model(P[:, perm], M, gm.relations(P[:, perm], None, M))
    assert torch.allclose(a, b, atol=1e-5)
