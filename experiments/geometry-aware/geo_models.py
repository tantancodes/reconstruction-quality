"""Set models over camera configurations, including geometry-aware attention.

Every network sees one camera configuration as a set of N cameras (padded to NMAX with a mask)
and predicts one number. Shapes below use B = batch, N = cameras, h = hidden width, H = heads,
d = h / H, R = number of pairwise relation features.

  pointnet     phi(x_i) per camera, max-pool, rho          (cameras never see each other)
  attention    the existing AttnSet: embed, one 4-head self-attention, residual, ff, max-pool
  geo          same as attention, plus a learned bias on the attention logits computed from
               rotation-invariant pairwise geometry (distance, angular separation, view agreement)
  geo_inv      geo, but the per-camera tokens get only rotation-invariant features, so the whole
               model is invariant to the arbitrary frame instead of learning it from augmentation
  geo_shuf     control: geo with the pairwise relations shuffled within each set, so it has the
               same parameters and compute as geo but the geometry no longer matches the pairs
"""
import math

import torch
import torch.nn as nn


# ------------------------------------------------------------------ geometry

def relations(P, V=None, mask=None):
    """Rotation-invariant pairwise relations for centred, scale-normalised camera positions.

    P (B, N, 3) positions, V (B, N, 3) unit view directions or None.
    Returns (B, N, N, R):
      d_ij   = |p_i - p_j|                    distance between cameras
      c_ij   = r_i . r_j, r = p / |p|          cosine of the angle between them, seen from the centroid
      u_ij   = v_i . v_j   (only if V given)   do they look the same way
    All three are unchanged by any rotation of the frame, and the positions are already scale-normalised.
    """
    d = torch.cdist(P, P)
    r = P / P.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c = r @ r.transpose(1, 2)
    feats = [d, c]
    if V is not None:
        feats.append(V @ V.transpose(1, 2))
    rel = torch.stack(feats, -1)
    if mask is not None:
        rel = rel * (mask[:, :, None] & mask[:, None, :])[..., None]
    return rel


def invariant_tokens(P, V=None):
    """Per-camera features that do not change under rotation: |p_i| and, with views, v_i . r_i."""
    n = P.norm(dim=-1, keepdim=True)
    if V is None:
        return n
    r = P / n.clamp_min(1e-9)
    return torch.cat([n, (V * r).sum(-1, keepdim=True)], -1)


# ------------------------------------------------------------------ layers

class GeoAttention(nn.Module):
    """Multi-head self-attention with an additive geometric bias.

    logits[b, k, i, j] = (q_i . k_j) / sqrt(d) + g_k(rel_ij)
    where g is a small MLP shared over pairs with one output per head. Without the bias term this
    is exactly scaled dot-product attention (same as nn.MultiheadAttention).
    """

    def __init__(self, h, heads, rel_dim, rel_hidden=16):
        super().__init__()
        assert h % heads == 0
        self.h, self.heads, self.d = h, heads, h // heads
        self.qkv = nn.Linear(h, 3 * h)
        self.out = nn.Linear(h, h)
        self.bias = nn.Sequential(nn.Linear(rel_dim, rel_hidden), nn.ReLU(), nn.Linear(rel_hidden, heads)) \
            if rel_dim else None

    def forward(self, f, mask, rel=None, return_weights=False):
        B, N, _ = f.shape
        q, k, v = self.qkv(f).view(B, N, 3, self.heads, self.d).permute(2, 0, 3, 1, 4)  # each (B, H, N, d)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.d)                               # (B, H, N, N)
        if self.bias is not None and rel is not None:
            logits = logits + self.bias(rel).permute(0, 3, 1, 2)                           # (B, N, N, H) -> (B, H, N, N)
        logits = logits.masked_fill(~mask[:, None, None, :], float('-inf'))                # padded cameras get no weight
        w = logits.softmax(-1)                                                             # rows sum to 1 over j
        out = (w @ v).transpose(1, 2).reshape(B, N, self.h)                                # concat heads
        out = self.out(out)
        return (out, w) if return_weights else out


class PointNetSet(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, mask, rel=None):
        return self.rho(self.phi(x).masked_fill(~mask[..., None], -1e9).max(1).values).squeeze(-1)


class AttnSet(nn.Module):
    """embed -> one self-attention block (residual) -> ff -> max-pool -> rho.

    rel_dim = 0 reproduces the existing AttnSet; rel_dim > 0 adds the geometric bias. The two have the
    same layers otherwise, so any difference comes from the bias (about 100 extra parameters)."""

    def __init__(self, in_dim, h, heads=4, rel_dim=0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.attn = GeoAttention(h, heads, rel_dim)
        self.ff = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, mask, rel=None):
        f = self.embed(x)
        f = self.ff(f + self.attn(f, mask, rel))
        return self.rho(f.masked_fill(~mask[..., None], -1e9).max(1).values).squeeze(-1)


def shuffle_pairs(rel, mask, gen):
    """Control: permute which camera each relation row/column belongs to, within each set.
    Keeps the distribution of relation values, destroys which pair they describe.
    Padded cameras sort to the end, so real cameras are only permuted among themselves."""
    B, N = mask.shape
    keys = torch.rand(B, N, generator=gen).masked_fill(~mask, 2.0)
    perm = keys.argsort(1)                                                  # (B, N)
    b = torch.arange(B)[:, None, None]
    return rel[b, perm[:, :, None], perm[:, None, :]]


def make_model(name, coord_dim, h, rel_dim):
    """coord_dim: per-camera input width for the frame-dependent models (3 = position, 6 = + view)."""
    if name == 'pointnet':
        return PointNetSet(coord_dim, h)
    if name == 'attention':
        return AttnSet(coord_dim, h, rel_dim=0)
    if name in ('geo', 'geo_shuf'):
        return AttnSet(coord_dim, h, rel_dim=rel_dim)
    if name == 'geo_inv':
        return AttnSet(1 if coord_dim == 3 else 2, h, rel_dim=rel_dim)
    raise ValueError(name)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
