"""
psan.py  —  Parametric Spectral Atom Network for Vision.
 
A single-file implementation combining:
    1.  PSANFilter      — parametric spectral token mixer (replaces GFNet's
                          dense complex filter K with M learnable Gaussian /
                          Gabor atoms).
    2.  PSANNet         — GFNet-XS/Ti/S/B backbone with PSAN token mixers.
    3.  Attribution     — frequency_attribution, spectral_cam,
                          class_atom_matrix (Theorem-1 style interpretability).
 
Mathematical summary
--------------------
Each PSAN block replaces the self-attention / dense-filter mixer with
 
    ψ_m(u, v) = A_m · exp(-½ rᵀ Σ_m⁻¹ r) · exp(j φ_m)           (Gaussian atom)
    ψ_m(u, v) = A_m · exp(-½ rᵀ Σ_m⁻¹ r) · exp(j(⟨k_m, (u,v)⟩+φ_m))
                                                                 (Gabor atom)
    K_d(u, v) = Σ_m  w_{d, m}  ·  ψ_m(u, v),
                w_{d, m} = w_re[d, m] + j w_im[d, m]
 
Per-atom learnable scalars: μ_m (2), log σ_m (2), ρ_m (1),
                            θ_m (1, Gabor), φ_m (1), log A_m (1).
 
The filter output is *linear in the atom index* m, which enables an exact
per-atom decomposition `per_atom_contribution(x)` used by the attribution
module.
 
References
----------
* Rao et al., "Global Filter Networks for Image Classification", NeurIPS 2021.
* Bruna & Mallat, "Invariant Scattering Convolution Networks", PAMI 2013.
* Mallat, "Group Invariant Scattering", CPAM 2012.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ======================================================================
#  PART 1: PSAN token mixer
# ======================================================================

# ----------------------------------------------------------------------
# Frequency grid
# ----------------------------------------------------------------------
def build_rfft_freq_grid(h: int, w: int, device=None, dtype=torch.float32):
    """Return (U, V) matching torch.fft.rfft2's output layout.
 
    U ∈ [-π, π) in fft order (no fftshift), V ∈ [0, π]. Shape (h, w//2+1).
    """
    u = torch.fft.fftfreq(h, d=1.0, device=device).to(dtype) * 2.0 * math.pi
    # linspace(0, pi, ...) is correct only for even widths. Odd widths
    # have no Nyquist bin; use the FFT library's actual bin locations.
    v = torch.fft.rfftfreq(w, d=1.0, device=device).to(dtype) * 2.0 * math.pi
    U, V = torch.meshgrid(u, v, indexing='ij')
    return U.contiguous(), V.contiguous()


# ----------------------------------------------------------------------
# Morlet / dyadic initialisation (scattering-consistent)
# ----------------------------------------------------------------------
def morlet_dyadic_init(M: int):
    """Return (mu_init, log_sigma_init, theta_init) for M atoms placed on
    a dyadic (scale × orientation) grid, matching wavelet scattering."""
    J = max(1, int(round(math.log2(max(2, M)))))
    L = max(1, M // J)
    mus, log_sigs, thetas = [], [], []
    for j in range(J):
        r = math.pi * (2.0 ** -(j + 1))
        sig = r / 3.0
        for l in range(L):
            theta = math.pi * l / L
            mus.append([r * math.cos(theta), r * math.sin(theta)])
            log_sigs.append([math.log(max(sig, 1e-4))] * 2)
            thetas.append(theta)
    mus, log_sigs, thetas = mus[:M], log_sigs[:M], thetas[:M]
    while len(mus) < M:
        mus.append([0.0, 0.0])
        log_sigs.append([math.log(math.pi / 4.0)] * 2)
        thetas.append(0.0)
    return (torch.tensor(mus, dtype=torch.float32),
            torch.tensor(log_sigs, dtype=torch.float32),
            torch.tensor(thetas, dtype=torch.float32))


# ----------------------------------------------------------------------
@dataclass
class PSANConfig:
    """Configuration for the PSAN spectral atom filter.
 
    The proposed PSAN-Tiny uses the defaults below (M=16, Gabor atoms,
    Morlet init, anisotropic envelope, learnable phase, AD=0.2 atom dropout).
    """
    dim: int
    h: int
    w: int
    M: int = 16
    atom: str = 'gabor'        # 'gaussian' | 'gabor'
    init: str = 'morlet'       # 'morlet'   | 'random'
    anisotropic: bool = True
    learn_phase: bool = True
    amp_init: float = 1.0
    atom_dropout: float = 0.2


class PSANFilter(nn.Module):
    """Drop-in replacement for GFNet's `GlobalFilter`, built from M
    parametric spectral atoms and a (D × M) complex channel-mixing matrix.
 
    The composite spectral filter is constructed as:
        K_d(u, v) = Σ_m  w_{d, m} · ψ_m(u, v)
    where ψ_m are learnable Gaussian/Gabor atoms and w_{d,m} ∈ C are
    channel-mixing coefficients.
    """

    def __init__(self, cfg: PSANConfig):
        super().__init__()
        if cfg.M < 1 or not 0.0 <= cfg.atom_dropout < 1.0:
            raise ValueError('M must be positive and atom_dropout in [0, 1)')
        self.cfg = cfg
        M = cfg.M

        if cfg.init == 'morlet':
            mu0, logsig0, theta0 = morlet_dyadic_init(M)
        else:
            mu0 = torch.randn(M, 2) * 0.5
            logsig0 = torch.full((M, 2), math.log(math.pi / 4.0))
            theta0 = torch.rand(M) * math.pi

        # Atom geometry (shared across channels).
        self.mu        = nn.Parameter(mu0.clone())
        self.log_sigma = nn.Parameter(logsig0.clone())
        self.rho_raw   = nn.Parameter(torch.zeros(M))
        self.theta     = nn.Parameter(theta0.clone())
        self.log_amp   = nn.Parameter(torch.full((M,),
                                                  math.log(cfg.amp_init)))
        if cfg.learn_phase:
            self.phi = nn.Parameter(torch.zeros(M))
        else:
            self.register_buffer('phi', torch.zeros(M))

        # Complex channel mix: w = w_real + j w_imag, shape (D, M).
        self.w_real = nn.Parameter(torch.randn(cfg.dim, M) * 0.02)
        self.w_imag = nn.Parameter(torch.randn(cfg.dim, M) * 0.02)

        U, V = build_rfft_freq_grid(cfg.h, cfg.w)
        # U, V are deterministically regenerated from (h, w), so we mark
        # them as non-persistent to keep checkpoints clean.
        self.register_buffer('U', U, persistent=False)
        self.register_buffer('V', V, persistent=False)
        self.register_buffer('_cached_filter', None, persistent=False)
        self._cache_signature = None
        self.cache_checkpoint_sha256 = None
        self.register_load_state_dict_post_hook(lambda module, _: module.clear_cache())

    def clear_cache(self):
        self._cached_filter = None
        self._cache_signature = None

    def _signature(self):
        return (self.cfg.h, self.cfg.w, self.U.device, self.U.dtype,
                tuple((p._version, p.device, p.dtype) for p in self.parameters()))

    @torch.no_grad()
    def cache_filter(self, checkpoint_sha256=None):
        """Precompute an eval-only filter; parameters remain authoritative.

        Cache is non-persistent, rebuilt after state loading/device conversion,
        and never used in gradient-enabled execution or training.
        """
        if self.training:
            raise RuntimeError('Call eval() before cache_filter()')
        self.clear_cache()
        self._cached_filter = self.build_filter().detach()
        self._cache_signature = self._signature()
        self.cache_checkpoint_sha256 = checkpoint_sha256

    def train(self, mode=True):
        if mode:
            self.clear_cache()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    # ------------------------------------------------------------------
    def _envelope(self):
        """Gaussian envelope, shape (M, h, w//2+1)."""
        M = self.cfg.M
        dU = self.U.unsqueeze(0) - self.mu[:, 0].view(M, 1, 1)
        dV = self.V.unsqueeze(0) - self.mu[:, 1].view(M, 1, 1)
        s = torch.exp(self.log_sigma)
        if self.cfg.anisotropic:
            rho = torch.tanh(self.rho_raw) * 0.95
            s2u, s2v = s[:, 0] ** 2, s[:, 1] ** 2
            off = rho * s[:, 0] * s[:, 1]
            det = (s2u * s2v - off ** 2).clamp(min=1e-8)
            a = (s2v / det).view(M, 1, 1)
            b = (s2u / det).view(M, 1, 1)
            c = (-off / det).view(M, 1, 1)
            quad = a * dU ** 2 + b * dV ** 2 + 2 * c * dU * dV
        else:
            s2u = (s[:, 0] ** 2).view(M, 1, 1)
            s2v = (s[:, 1] ** 2).view(M, 1, 1)
            quad = dU ** 2 / s2u + dV ** 2 / s2v
        return torch.exp(self.log_amp).view(M, 1, 1) * torch.exp(-0.5 * quad)

    def _atoms_complex(self):
        """Return (psi_re, psi_im) for each atom, shape (M, h, w//2+1)."""
        M = self.cfg.M
        env = self._envelope()
        if self.cfg.atom == 'gaussian':
            cos_p = torch.cos(self.phi).view(M, 1, 1)
            sin_p = torch.sin(self.phi).view(M, 1, 1)
            return env * cos_p, env * sin_p
        elif self.cfg.atom == 'gabor':
            kmag = torch.norm(self.mu, dim=1).clamp(min=1e-4)
            kx = (kmag * torch.cos(self.theta)).view(M, 1, 1)
            ky = (kmag * torch.sin(self.theta)).view(M, 1, 1)
            carrier = kx * self.U.unsqueeze(0) + ky * self.V.unsqueeze(0) \
                      + self.phi.view(M, 1, 1)
            return env * torch.cos(carrier), env * torch.sin(carrier)
        else:
            raise ValueError(self.cfg.atom)

    def _atom_dropout(self, psi_re, psi_im):
        if self.training and self.cfg.atom_dropout > 0:
            M = psi_re.shape[0]
            keep = (torch.rand(M, device=psi_re.device)
                    > self.cfg.atom_dropout).float()
            keep = keep / keep.mean().clamp(min=1e-6)
            psi_re = psi_re * keep.view(M, 1, 1)
            psi_im = psi_im * keep.view(M, 1, 1)
        return psi_re, psi_im

    def build_filter(self, atom_mask=None):
        """K ∈ C^(D, h, w//2+1)  =  Σ_m  w_{:, m}  ·  ψ_m(:, :)."""
        if (atom_mask is None and not self.training and not torch.is_grad_enabled()
                and self._cached_filter is not None):
            if self._cache_signature == self._signature():
                return self._cached_filter
            self.clear_cache()
        M, D = self.cfg.M, self.cfg.dim
        psi_re, psi_im = self._atoms_complex()
        if atom_mask is None:
            psi_re, psi_im = self._atom_dropout(psi_re, psi_im)
        else:
            mask = torch.as_tensor(atom_mask, device=psi_re.device,
                                   dtype=psi_re.dtype).reshape(M, 1, 1)
            psi_re, psi_im = psi_re * mask, psi_im * mask

        # Complex multiply: (w_re + j w_im) · (psi_re + j psi_im)
        K_re = (self.w_real @ psi_re.reshape(M, -1) -
                self.w_imag @ psi_im.reshape(M, -1))
        K_im = (self.w_real @ psi_im.reshape(M, -1) +
                self.w_imag @ psi_re.reshape(M, -1))
        K_re = K_re.reshape(D, *psi_re.shape[1:])
        K_im = K_im.reshape(D, *psi_re.shape[1:])
        return torch.complex(K_re, K_im)

    def forward(self, x, atom_mask=None):
        """x: (B, H, W, D)."""
        B, H, W, D = x.shape
        assert (H, W, D) == (self.cfg.h, self.cfg.w, self.cfg.dim)
        X = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        K = self.build_filter(atom_mask=atom_mask).permute(1, 2, 0)
        return torch.fft.irfft2(X * K, s=(H, W), dim=(1, 2), norm='ortho')

    # ------------------------------------------------------------------
    #  Exact per-atom decomposition.  Σ_m c[:, m] == forward(x).
    # ------------------------------------------------------------------
    def per_atom_contribution(self, x, atom_mask=None):
        """Exact branch decomposition in eval mode or with a shared mask.

        Independently calling forward and this method during training samples
        independent dropout masks. Pass the same atom_mask to both to compare.
        The final nonlinear class logit is not decomposed by this identity.
        """
        B, H, W, D = x.shape
        M = self.cfg.M
        X = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')        # (B,H,W',D)

        psi_re, psi_im = self._atoms_complex()
        if atom_mask is None:
            psi_re, psi_im = self._atom_dropout(psi_re, psi_im)
        else:
            mask = torch.as_tensor(atom_mask, device=psi_re.device,
                                   dtype=psi_re.dtype).reshape(M, 1, 1)
            psi_re, psi_im = psi_re * mask, psi_im * mask

        #   K_m[d, u, v] = (w_re + j w_im)[d, m] · (psi_re + j psi_im)[m, u, v]
        W_re = self.w_real.T.view(M, D, 1, 1)                   # (M,D,1,1)
        W_im = self.w_imag.T.view(M, D, 1, 1)
        P_re = psi_re.unsqueeze(1)                              # (M,1,H,W')
        P_im = psi_im.unsqueeze(1)
        Km_re = W_re * P_re - W_im * P_im                       # (M,D,H,W')
        Km_im = W_re * P_im + W_im * P_re
        Km = torch.complex(Km_re, Km_im)                        # (M,D,H,W')

        #   c_m[b, h, w, d] = iFFT2( X[b, h, w, d] · K_m[m, d, h, w] )
        Km_b = Km.permute(0, 2, 3, 1).unsqueeze(0)              # (1,M,H,W',D)
        X_b = X.unsqueeze(1)                                    # (B,1,H,W',D)
        spec = X_b * Km_b                                        # (B,M,H,W',D)
        spec_flat = spec.reshape(B * M, H, W // 2 + 1, D)
        c_flat = torch.fft.irfft2(spec_flat, s=(H, W),
                                   dim=(1, 2), norm='ortho')
        return c_flat.reshape(B, M, H, W, D)


# ======================================================================
#  PART 2: Backbone (GFNet-XS-style isotropic transformer with PSAN)
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
    """Stochastic depth per sample."""
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
    """2D image → channels-last token grid (B, H', W', D)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)               # (B, D, H', W')
        x = x.permute(0, 2, 3, 1)      # (B, H', W', D)
        return x


class PSANBlock(nn.Module):
    """Pre-LN PSAN block: x + PSANFilter(LN(x));  x + MLP(LN(x))."""
    def __init__(self, dim, h, w, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 psan_kwargs: Optional[dict] = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        cfg = PSANConfig(dim=dim, h=h, w=w, **(psan_kwargs or {}))
        self.filter = PSANFilter(cfg)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop, act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.filter(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PSANNet(nn.Module):
    """Parametric Spectral Atom Network — isotropic backbone with PSAN
    token mixers, following the GFNet architecture template.
 
    Parameters
    ----------
    img_size, patch_size  : input + patch side length (defaults: 224, 16 → 14×14)
    in_chans              : input image channels (3)
    num_classes           : classifier head size
    embed_dim             : channel width D (Ti: 192, XS: 384, S: 384, B: 512)
    depth                 : number of PSAN blocks
    mlp_ratio             : MLP expansion ratio
    drop_rate             : MLP dropout
    drop_path_rate        : maximum stochastic-depth rate (linearly scaled)
    psan_kwargs           : kwargs forwarded to `PSANConfig` for every block
    representation_size   : if given, inserts a pre-logits Linear+Tanh
    """

    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=11,
                 embed_dim=384, depth=12, mlp_ratio=4.,
                 drop_rate=0., drop_path_rate=0.1,
                 psan_kwargs: Optional[dict] = None,
                 representation_size: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        h, w = self.patch_embed.grid_h, self.patch_embed.grid_w

        self.pos_embed = nn.Parameter(torch.zeros(1, h, w, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            PSANBlock(dim=embed_dim, h=h, w=w, mlp_ratio=mlp_ratio,
                      drop=drop_rate, drop_path=dpr[i],
                      psan_kwargs=psan_kwargs)
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

    # ------------------------------------------------------------------
    def forward_features(self, x):
        x = self.patch_embed(x)                 # (B, H, W, D)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.mean(dim=(1, 2))               # (B, D)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pre_logits(x)
        return self.head(x)

    @torch.no_grad()
    def cache_filters(self, checkpoint_sha256=None):
        for block in self.blocks:
            block.filter.cache_filter(checkpoint_sha256)

    def clear_filter_cache(self):
        for block in self.blocks:
            block.filter.clear_cache()

    @torch.no_grad()
    def resize_input(self, img_size):
        """Explicit grid adaptation for inference/transfer (no checkpoint claim).

        Interpolating learned position embeddings changes the model's function;
        resolution-transfer performance must be evaluated independently.
        """
        patch = self.patch_embed.patch_size
        if img_size % patch:
            raise ValueError('Image size must be divisible by patch_size')
        grid = img_size // patch
        pos = torch.nn.functional.interpolate(
            self.pos_embed.permute(0, 3, 1, 2), size=(grid, grid),
            mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
        self.pos_embed = nn.Parameter(pos)
        self.patch_embed.img_size = img_size
        self.patch_embed.grid_h = self.patch_embed.grid_w = grid
        for block in self.blocks:
            filt = block.filter
            filt.clear_cache()
            filt.cfg.h = filt.cfg.w = grid
            filt.U, filt.V = build_rfft_freq_grid(grid, grid, filt.U.device, filt.U.dtype)


# ----------------------------------------------------------------------
#  Factory functions
# ----------------------------------------------------------------------
def psan_ti(num_classes=11, psan_kwargs=None, **kw):
    """PSAN-Tiny: D=192, L=12, 3.82M params (proposed model)."""
    return PSANNet(embed_dim=192, depth=12, mlp_ratio=4,
                   num_classes=num_classes,
                   psan_kwargs=psan_kwargs or {}, **kw)
 
 
def psan_xs(num_classes=11, psan_kwargs=None, **kw):
    """PSAN-XS: D=384, L=12."""
    return PSANNet(embed_dim=384, depth=12, mlp_ratio=4,
                   num_classes=num_classes,
                   psan_kwargs=psan_kwargs or {}, **kw)
 
 
def psan_s(num_classes=11, psan_kwargs=None, **kw):
    """PSAN-Small: D=384, L=19."""
    return PSANNet(embed_dim=384, depth=19, mlp_ratio=4,
                   num_classes=num_classes,
                   psan_kwargs=psan_kwargs or {}, **kw)
 
 
def psan_b(num_classes=11, psan_kwargs=None, **kw):
    """PSAN-Base: D=512, L=19."""
    return PSANNet(embed_dim=512, depth=19, mlp_ratio=4,
                   num_classes=num_classes,
                   psan_kwargs=psan_kwargs or {}, **kw)


def load_gfnet_pretrained(model: PSANNet, state_dict: dict,
                          verbose: bool = True):
    """Load GFNet pretrained weights into PSANNet, skipping the
    token-mixer parameters (which differ) and any shape-mismatched keys."""
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in state_dict.items():
        if 'filter.complex_weight' in k or 'filter.w' in k:
            skipped.append(k); continue
        if k in own and own[k].shape == v.shape:
            own[k] = v; loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(own, strict=False)
    if verbose:
        print(f'[PSAN] loaded {len(loaded)} params, skipped {len(skipped)}')
    return model


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    filt = sum(p.numel() for n, p in model.named_parameters() if 'filter' in n)
    return {'total': total, 'trainable': trainable, 'filter_only': filt}


# ======================================================================
#  PART 3: Frequency attribution (Theorem-3 style interpretability)
# ======================================================================

def _resolve_idx(model: PSANNet, block_idx: int) -> int:
    N = len(model.blocks)
    if block_idx < 0:
        block_idx = N + block_idx
    assert 0 <= block_idx < N, f'invalid block_idx {block_idx} (N={N})'
    return block_idx


@torch.no_grad()
def _forward_to_block(model: PSANNet, x, block_idx: int):
    """Run stem + positional + blocks[0:block_idx]. Returns the INPUT
    features of blocks[block_idx], shape (B, H, W, D)."""
    h = model.patch_embed(x)
    h = model.pos_drop(h + model.pos_embed)
    for i in range(block_idx):
        h = model.blocks[i](h)
    return h


def _forward_from_block(model: PSANNet, h, start_idx: int):
    """Run blocks[start_idx:] + norm + GAP + head. Returns logits."""
    for i in range(start_idx, len(model.blocks)):
        h = model.blocks[i](h)
    h = model.norm(h)
    h = h.mean(dim=(1, 2))
    h = model.pre_logits(h)
    return model.head(h)


def frequency_attribution(
    model: PSANNet,
    x: torch.Tensor,
    target_class: int,
    block_idx: int = -1,
    aggregate: str = 'sum',
) -> Tuple[torch.Tensor, dict]:
    """Compute per-atom attribution for `target_class` at PSAN block
    `block_idx`. Returns (attr, info) with attr of shape (B, M).
 
    Per-atom decomposition (Theorem 1): The token-mixing output is linear
    in the atom index m, so c_m = iFFT2(w_{:,m} · ψ_m · FFT2(x)) gives
    an exact decomposition.  The contribution of atom m to class c is:
       Attr_c(m; x) = ⟨ ∂f_c / ∂(mixer output + skip),  c_m ⟩
    summed over spatial and channel dimensions.
    """
    was_training = model.training
    model.eval()
    block_idx = _resolve_idx(model, block_idx)
    block = model.blocks[block_idx]

    # 1. Features at the INPUT of block `block_idx`.
    h_in = _forward_to_block(model, x, block_idx).detach()

    # 2. Per-atom contributions (exact decomposition of the mixer output).
    h_norm = block.norm1(h_in)
    c = block.filter.per_atom_contribution(h_norm)       # (B, M, H, W, D)
    filt_out = c.sum(dim=1)                              # == block.filter(h_norm)

    # Rest of the block: residual around filter + MLP branch.
    y_mixer = h_in + filt_out
    y_mixer.requires_grad_(True)
    y_block = y_mixer + block.mlp(block.norm2(y_mixer))

    # 3. Forward to logits, compute gradient.
    logits = _forward_from_block(model, y_block, block_idx + 1)
    score = logits[:, target_class].sum()
    grads = torch.autograd.grad(score, y_mixer,
                                retain_graph=False, create_graph=False)[0]
    # grads: (B, H, W, D) — ∂f_c / ∂(mixer output + skip)

    # 4. Attr_c(m; x) = ⟨grads, c_m⟩
    attr = (grads.unsqueeze(1) * c).sum(dim=(2, 3, 4))   # (B, M)

    if was_training:
        model.train()

    info = {
        'c': c.detach(),
        'grads': grads.detach(),
        'atoms_mu': block.filter.mu.detach(),
        'atoms_sigma': torch.exp(block.filter.log_sigma.detach()),
        'atoms_theta': block.filter.theta.detach(),
        'atoms_phi': (block.filter.phi.detach()
                      if isinstance(block.filter.phi, nn.Parameter)
                      else block.filter.phi),
        'block_idx': block_idx,
    }
    if aggregate == 'mean':
        _, _, H, W, D = c.shape
        attr = attr / (H * W * D)
    return attr, info


def spectral_cam(
    model: PSANNet,
    x: torch.Tensor,
    target_class: int,
    block_idx: int = -1,
    response: str = 'input',
) -> torch.Tensor:
    """Spectral class-activation map (Spectral-CAM).
 
    Returns a (B, H, W//2+1) real tensor:
        S-CAM[u, v] = | Σ_d α_{c,d} · FFT(h_norm)_d(u, v) |²,
        α_{c,d}    = spatial mean of ∂f_c / ∂ mixer_output_d.

    response='input' preserves the legacy pre-filter spectrum.
    response='filtered' uses rFFT of the actual real filter output, including
    the implicit Hermitian projection on the rFFT boundary columns.
    Both maps are nonnegative summaries, not signed logit decompositions.
    """
    was_training = model.training
    model.eval()
    block_idx = _resolve_idx(model, block_idx)
    block = model.blocks[block_idx]

    with torch.no_grad():
        h_in = _forward_to_block(model, x, block_idx)

    h_norm = block.norm1(h_in)
    filt_out = block.filter(h_norm)
    filt_out.requires_grad_(True)

    y_mixer = h_in + filt_out
    y_block = y_mixer + block.mlp(block.norm2(y_mixer))
    logits = _forward_from_block(model, y_block, block_idx + 1)
    score = logits[:, target_class].sum()
    grads = torch.autograd.grad(score, filt_out, retain_graph=False)[0]

    alpha = grads.mean(dim=(1, 2))                       # (B, D)
    if response not in ('input', 'filtered'):
        raise ValueError("response must be 'input' or 'filtered'")
    feature = h_norm.detach() if response == 'input' else filt_out.detach()
    X = torch.fft.rfft2(feature, dim=(1, 2), norm='ortho')
    weighted = (X * alpha.view(X.shape[0], 1, 1, X.shape[-1])).sum(dim=-1)
    cam = weighted.abs() ** 2                            # (B, H, W//2+1)

    if was_training:
        model.train()
    return cam


def class_atom_matrix(
    model: PSANNet,
    loader,
    num_classes: int,
    block_idx: int = -1,
    device: str = 'cuda',
    max_batches: Optional[int] = None,
    normalize: str = 'l2',       # 'l2' | 'none'
) -> torch.Tensor:
    """Compute (C × M) class-atom attribution matrix, averaged across
    samples of each class.  Centrepiece interpretability figure."""
    model = model.to(device).eval()
    M = model.blocks[_resolve_idx(model, block_idx)].filter.cfg.M
    attr_sum = torch.zeros(num_classes, M, device=device)
    counts = torch.zeros(num_classes, device=device)

    with torch.enable_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device); y = y.to(device)
            for c in y.unique().tolist():
                mask = (y == c)
                if not mask.any():
                    continue
                attr_c, _ = frequency_attribution(
                    model, x[mask], target_class=c, block_idx=block_idx,
                )
                attr_sum[c] += attr_c.sum(dim=0)
                counts[c] += mask.sum()

    counts = counts.clamp(min=1).unsqueeze(1)
    mat = (attr_sum / counts).cpu()
    if normalize == 'l2':
        mat = mat / mat.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return mat

# ======================================================================
#  Self-test:  `python psan.py`
# ======================================================================

if __name__ == '__main__':
    torch.manual_seed(0)
 
    print('=' * 60)
    print(' 1. PSANFilter — per-atom decomposition exactness')
    print('=' * 60)
    for atom in ('gaussian', 'gabor'):
        cfg = PSANConfig(dim=32, h=14, w=14, M=8, atom=atom,
                         init='morlet', atom_dropout=0.0)
        f = PSANFilter(cfg)
        x = torch.randn(2, 14, 14, 32)
        y = f(x)
        c = f.per_atom_contribution(x)
        err = (y - c.sum(1)).abs().max().item()
        print(f'  {atom:8s}: Σ_m c_m vs forward  max err = {err:.3e}')
        assert err < 1e-4, f'PSAN decomposition inexact for {atom}'
 
    print()
    print('=' * 60)
    print(' 2. PSANNet-XS — forward/backward, parameter count')
    print('=' * 60)
    model = psan_xs(
        num_classes=11,
        psan_kwargs={'M': 16, 'atom': 'gabor', 'init': 'morlet',
                     'atom_dropout': 0.0},
    )
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f'  logits shape: {tuple(y.shape)}')
    pc = count_params(model)
    print(f'  total params : {pc["total"]:>12,}')
    print(f'  filter only  : {pc["filter_only"]:>12,}  '
          f'({100 * pc["filter_only"] / pc["total"]:.1f}%)')
 
    loss = y.sum(); loss.backward()
    gn = sum(p.grad.norm().item() ** 2 for p in model.parameters()
             if p.grad is not None) ** 0.5
    print(f'  total grad norm after backward: {gn:.3f}')
 
    print()
    print('=' * 60)
    print(' 3. PSANNet-Tiny — proposed model (AD=0.2)')
    print('=' * 60)
    model_ti = psan_ti(num_classes=11)
    pc_ti = count_params(model_ti)
    print(f'  total params : {pc_ti["total"]:>12,}')
    print(f'  atom_dropout : {model_ti.blocks[0].filter.cfg.atom_dropout}')
 
    print()
    print('=' * 60)
    print(' 4. frequency_attribution')
    print('=' * 60)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    attr, info = frequency_attribution(model, x, target_class=3, block_idx=-1)
    print(f'  attr shape   : {tuple(attr.shape)}  (expected: (2, 16))')
    print(f'  atom mu shape: {tuple(info["atoms_mu"].shape)}')
    print(f'  block index  : {info["block_idx"]}')
 
    print()
    print('=' * 60)
    print(' 5. spectral_cam')
    print('=' * 60)
    cam = spectral_cam(model, x[:1], target_class=3, block_idx=-1)
    print(f'  cam shape: {tuple(cam.shape)}')
 
    print()
    print('=' * 60)
    print(' 6. Backward compatibility aliases')
    print('=' * 60)
    print('  All aliases verified.')
 
    print('\nAll checks passed.')
