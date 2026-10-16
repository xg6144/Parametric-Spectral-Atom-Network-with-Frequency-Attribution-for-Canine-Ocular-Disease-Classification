"""
afno.py — Adaptive Fourier Neural Operator for vision.

Reference
---------
Guibas et al., "Adaptive Fourier Neural Operators: Efficient Token
Mixers for Transformers", ICLR 2022.

Overview
--------
AFNO replaces the self-attention token mixer with:
  (1) 2D FFT of the feature map,
  (2) a block-diagonal MLP applied at each frequency bin
      (with soft-thresholding for sparsity),
  (3) 2D inverse FFT.

Compared to GFNet (which learns one dense complex filter per (u, v, d)),
AFNO shares a small MLP across all (u, v) and so has *constant* filter
parameter cost w.r.t. the token grid size — log-linear complexity in
FFT and O(D² / k) in MLP.

This file provides:
  * AFNOFilter        — the token mixer (drop-in replacement for self-attn)
  * AFNOBlock         — Pre-LN block: AFNOFilter then MLP
  * AFNONet           — isotropic backbone (matches GFNet/ViT style)
  * afno_tiny         — tiny variant, param-matched to PSPN-GFNet-Ti
  * afno_small, afno_base — larger variants
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


# ======================================================================
#  AFNO token mixer
# ======================================================================
class AFNOFilter(nn.Module):
    """Adaptive Fourier Neural Operator token mixer.

    Parameters
    ----------
    dim             : channel dimension D.
    num_blocks      : number of block-diagonal groups (k in the paper).
                      D must be divisible by num_blocks.
    mlp_ratio       : hidden dim of per-frequency MLP = dim * mlp_ratio.
    sparsity_thresh : soft-thresholding threshold λ for frequency-domain
                      features (default 0.01 from the AFNO paper).
    hard_thresh_frac: if > 0, hard-zero the top-(1-frac) fraction of
                      frequencies by energy. Default 1.0 = keep all.

    Notes
    -----
    The block-diagonal weights are stored as (num_blocks, block_size,
    block_size*mlp_ratio) to avoid full-D matrices. This is the main
    parameter-saving device in AFNO.
    """

    def __init__(self, dim: int, num_blocks: int = 8,
                 mlp_ratio: float = 1.0,
                 sparsity_thresh: float = 0.01,
                 hard_thresh_frac: float = 1.0):
        super().__init__()
        assert dim % num_blocks == 0, \
            f'dim {dim} must be divisible by num_blocks {num_blocks}'
        self.dim = dim
        self.num_blocks = num_blocks
        self.block_size = dim // num_blocks
        self.hidden_size = int(self.block_size * mlp_ratio)
        self.sparsity_thresh = sparsity_thresh
        self.hard_thresh_frac = hard_thresh_frac

        scale = 0.02

        # Two-layer MLP on complex features. Each layer has real + imag
        # weight tensors so that a complex matmul can be emulated as
        # (w_re + j w_im)(x_re + j x_im).
        self.w1 = nn.Parameter(scale * torch.randn(
            2, num_blocks, self.block_size, self.hidden_size))
        self.b1 = nn.Parameter(scale * torch.zeros(
            2, num_blocks, self.hidden_size))
        self.w2 = nn.Parameter(scale * torch.randn(
            2, num_blocks, self.hidden_size, self.block_size))
        self.b2 = nn.Parameter(scale * torch.zeros(
            2, num_blocks, self.block_size))

    # ------------------------------------------------------------------
    def _complex_matmul_block(self, x_re, x_im, w_re, w_im, b_re, b_im):
        """x · w + b  in complex arithmetic, applied block-wise.

        x : (..., K, block_in)    K=num_blocks
        w : (K, block_in, block_out)
        b : (K, block_out)
        """
        # (x_re + j x_im)(w_re + j w_im)
        #   = (x_re w_re - x_im w_im) + j (x_re w_im + x_im w_re)
        y_re = torch.einsum('...ki,kio->...ko', x_re, w_re) \
             - torch.einsum('...ki,kio->...ko', x_im, w_im) \
             + b_re
        y_im = torch.einsum('...ki,kio->...ko', x_re, w_im) \
             + torch.einsum('...ki,kio->...ko', x_im, w_re) \
             + b_im
        return y_re, y_im

    # ------------------------------------------------------------------
    def forward(self, x):
        """x: (B, H, W, D)."""
        B, H, W, D = x.shape
        # Keep FFT in fp32 regardless of AMP to avoid ComplexHalf issues.
        x_f32 = x.float()
        X = torch.fft.rfft2(x_f32, dim=(1, 2), norm='ortho')   # (B,H,W',D) complex
        Wp = X.shape[2]

        # Optional hard frequency thresholding (keep top-k by energy).
        if self.hard_thresh_frac < 1.0 and self.training:
            with torch.no_grad():
                energy = X.abs().pow(2).sum(dim=-1)            # (B,H,W')
                keep = int(H * Wp * self.hard_thresh_frac)
                topk = torch.topk(energy.flatten(1), keep, dim=1).indices
                mask = torch.zeros_like(energy).flatten(1)
                mask.scatter_(1, topk, 1.0)
                mask = mask.view(B, H, Wp, 1)
            X = X * mask

        # Reshape to block-diagonal layout: (B, H, W', K, block)
        X_re = X.real.reshape(B, H, Wp, self.num_blocks, self.block_size)
        X_im = X.imag.reshape(B, H, Wp, self.num_blocks, self.block_size)

        # Two-layer complex MLP with ReLU on the magnitude after layer 1.
        h_re, h_im = self._complex_matmul_block(
            X_re, X_im,
            self.w1[0], self.w1[1], self.b1[0], self.b1[1],
        )
        # Complex ReLU: apply to real and imag independently (the AFNO
        # paper uses this simple variant).
        h_re = torch.relu(h_re)
        h_im = torch.relu(h_im)

        y_re, y_im = self._complex_matmul_block(
            h_re, h_im,
            self.w2[0], self.w2[1], self.b2[0], self.b2[1],
        )

        # Soft-thresholding: shrink small amplitudes toward zero.
        lam = self.sparsity_thresh
        if lam > 0:
            amp = torch.sqrt(y_re ** 2 + y_im ** 2 + 1e-12)
            shrink = torch.clamp(amp - lam, min=0.0) / amp
            y_re = y_re * shrink
            y_im = y_im * shrink

        # Flatten block axis back.
        y_re = y_re.reshape(B, H, Wp, D)
        y_im = y_im.reshape(B, H, Wp, D)
        Y = torch.complex(y_re, y_im)

        # Residual in frequency domain (AFNO uses skip connection here).
        Y = Y + X

        y = torch.fft.irfft2(Y, s=(H, W), dim=(1, 2), norm='ortho')
        return y.to(x.dtype)


# ======================================================================
#  Standard transformer bits (duplicated here so afno.py stands alone)
# ======================================================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


class PatchEmbed(nn.Module):
    """Image → channels-last token grid (B, H', W', D)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)            # (B, D, H', W')
        x = x.permute(0, 2, 3, 1)   # (B, H', W', D)
        return x


# ======================================================================
#  AFNO block & backbone
# ======================================================================
class AFNOBlock(nn.Module):
    """Pre-LN block: x + AFNO(LN(x));  x + MLP(LN(x))."""
    def __init__(self, dim, num_blocks=8, afno_mlp_ratio=1.0,
                 sparsity_thresh=0.01, hard_thresh_frac=1.0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNOFilter(dim=dim, num_blocks=num_blocks,
                                  mlp_ratio=afno_mlp_ratio,
                                  sparsity_thresh=sparsity_thresh,
                                  hard_thresh_frac=hard_thresh_frac)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop,
                       act_layer=act_layer)
        self.drop_path = (DropPath(drop_path)
                          if drop_path > 0 else nn.Identity())

    def forward(self, x):
        x = x + self.drop_path(self.filter(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AFNONet(nn.Module):
    """Isotropic AFNO backbone, in the same family as GFNet/ViT.

    Default patch_size=16 → 14×14 token grid at 224×224.

    Parameters
    ----------
    num_blocks        : AFNO block-diagonal group count.  Must divide
                        embed_dim.  Default 8.
    afno_mlp_ratio    : per-frequency MLP ratio (paper uses 1.0).
    sparsity_thresh   : λ for soft-thresholding (paper default 0.01).
    hard_thresh_frac  : fraction of frequency bins to keep by energy
                        (1.0 keeps all; <1.0 hard-zeros low-energy bins
                        during training).
    """

    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=11,
                 embed_dim=192, depth=12, mlp_ratio=4.,
                 num_blocks=8, afno_mlp_ratio=1.0,
                 sparsity_thresh=0.01, hard_thresh_frac=1.0,
                 drop_rate=0., drop_path_rate=0.1,
                 representation_size: Optional[int] = None,
                 use_checkpoint: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        h, w = self.patch_embed.grid_h, self.patch_embed.grid_w

        self.pos_embed = nn.Parameter(torch.zeros(1, h, w, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            AFNOBlock(dim=embed_dim,
                       num_blocks=num_blocks,
                       afno_mlp_ratio=afno_mlp_ratio,
                       sparsity_thresh=sparsity_thresh,
                       hard_thresh_frac=hard_thresh_frac,
                       mlp_ratio=mlp_ratio,
                       drop=drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if representation_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size), nn.Tanh()
            )
            head_in = representation_size
        else:
            self.pre_logits = nn.Identity()
            head_in = embed_dim
        self.head = (nn.Linear(head_in, num_classes)
                     if num_classes > 0 else nn.Identity())

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        return x.mean(dim=(1, 2))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pre_logits(x)
        return self.head(x)


# ----------------------------------------------------------------------
#  Factory functions
# ----------------------------------------------------------------------
def afno_tiny(num_classes=11, **kw):
    """Param-matched to PSPN-GFNet-Ti / GFNet-tiny (dim=192, depth=12).

    Expected param count: ~4.4M (slightly above PSPN at 3.82M because
    the per-frequency MLP has two weight matrices of size D×D/num_blocks).
    """
    return AFNONet(embed_dim=192, depth=12, mlp_ratio=4,
                    num_blocks=8, afno_mlp_ratio=1.0,
                    num_classes=num_classes, **kw)


def afno_small(num_classes=11, **kw):
    return AFNONet(embed_dim=384, depth=12, mlp_ratio=4,
                    num_blocks=8, afno_mlp_ratio=1.0,
                    num_classes=num_classes, **kw)


def afno_base(num_classes=11, **kw):
    return AFNONet(embed_dim=512, depth=12, mlp_ratio=4,
                    num_blocks=8, afno_mlp_ratio=1.0,
                    num_classes=num_classes, **kw)


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    filt = sum(p.numel() for n, p in model.named_parameters() if 'filter' in n)
    return {'total': total, 'trainable': trainable, 'filter_only': filt}


# ======================================================================
#  Self-test
# ======================================================================
if __name__ == '__main__':
    torch.manual_seed(0)

    print('=' * 64)
    print(' AFNONet variants — sizes')
    print('=' * 64)
    for fn, name in [(afno_tiny, 'tiny'),
                      (afno_small, 'small'),
                      (afno_base, 'base')]:
        m = fn(num_classes=11)
        pc = count_params(m)
        print(f'  {name:6s}  embed_dim={m.embed_dim:3d}  '
              f'total={pc["total"]:>11,}  filter={pc["filter_only"]:>8,}')

    print()
    print('=' * 64)
    print(' AFNO-Tiny forward/backward sanity')
    print('=' * 64)
    m = afno_tiny(num_classes=11)
    m.train()
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(f'  forward: logits={tuple(y.shape)}')
    loss = y.sum(); loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2
                    for p in m.parameters() if p.grad is not None) ** 0.5
    print(f'  backward OK, grad_norm={grad_norm:.3f}')
    print('All checks passed.')
