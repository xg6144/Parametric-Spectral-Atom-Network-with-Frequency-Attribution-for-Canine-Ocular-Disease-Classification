from typing import Optional

import torch
import torch.nn as nn

class GlobalFilter(nn.Module):
    def __init__(
        self,
        dim: int,
        h: int,
        w: int,
        filter_init_std: float = 0.02,
    ):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w

        w_half = w // 2 + 1
        self.filter_real = nn.Parameter(
            torch.randn(h, w_half, dim) * filter_init_std
        )
        self.filter_imag = nn.Parameter(
            torch.randn(h, w_half, dim) * filter_init_std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert H == self.h and W == self.w and C == self.dim, (
            f"GlobalFilter expected (_, {self.h}, {self.w}, {self.dim}) "
            f"but got {tuple(x.shape)}"
        )

        x_hat = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        w_filter = torch.complex(self.filter_real, self.filter_imag)
        z_hat = x_hat * w_filter
        z = torch.fft.irfft2(z_hat, s=(H, W), dim=(1, 2), norm="ortho")
        return z

class FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


class GFNetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        h: int,
        w: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = GlobalFilter(dim=dim, h=h, w=w)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

class ConvStem(nn.Module):
    def __init__(self, in_chans: int = 3, out_dim: int = 192, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.proj(x)                          # (B, C, H', W')
        x = x.permute(0, 2, 3, 1).contiguous()    # (B, H', W', C)
        x = self.norm(x)
        return x

class GFNet(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 192,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        grid_size = img_size // patch_size
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.stem = ConvStem(in_chans=in_chans, out_dim=embed_dim, patch_size=patch_size)

        dpr = [float(x) for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            GFNetBlock(
                dim=embed_dim,
                h=grid_size,
                w=grid_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=(1, 2))
        return self.head(x)

def gfnet_tiny(num_classes: int = 1000, img_size: int = 224, **kwargs) -> GFNet:
    return GFNet(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


if __name__ == "__main__":
    print("\n[gfnet_tiny @ 224] checking forward / backward / param count")
    m = gfnet_tiny(num_classes=11)
    n = sum(p.numel() for p in m.parameters())
    print(f"  params: {n:,} ({n / 1e6:.2f}M)")
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(f"  forward: {tuple(x.shape)} -> {tuple(y.shape)}")
    y.sum().backward()
    for name, p in m.named_parameters():
        if p.grad is None:
            print(f"  [WARN] no grad: {name}")
    print("  backward: OK")
