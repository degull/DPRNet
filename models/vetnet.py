# 손 (Restoration Network) - VETNet
# G:\DPR-Net\models\vetnet.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.film import FiLMBlock

# -----------------------------------------------
# Volterra Layer (2D)
# -----------------------------------------------
def circular_shift(x, shift_x, shift_y):
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))

class VolterraLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=2, use_lossless=False):
        super().__init__()
        self.rank = rank
        self.use_lossless = use_lossless

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        if use_lossless:
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            P = kernel_size // 2
            self.shifts = [(dx, dy) for dx in range(-P, P+1) for dy in range(-P, P+1)
                           if not (dx == 0 and dy == 0) and (dx, dy) >= (0, 0)]
        else:
            self.W2a = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                                      for _ in range(rank)])
            self.W2b = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                                      for _ in range(rank)])

    def forward(self, x):
        linear = self.conv1(x)
        quad = 0

        if self.use_lossless:
            for dx, dy in self.shifts:
                x_s = circular_shift(x, dx, dy)
                quad += self.conv2(torch.clamp(x * x_s, -1, 1))
        else:
            for a, b in zip(self.W2a, self.W2b):
                qa = torch.clamp(a(x), -1, 1)
                qb = torch.clamp(b(x), -1, 1)
                quad += qa * qb

        return linear + quad


# -----------------------------------------------
# MDTA + GDFN + Volterra Transformer Block
# -----------------------------------------------
class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))
    def forward(self, x):
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return x / torch.sqrt(var + 1e-5) * self.weight

class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1))
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5) * self.weight + self.bias

def LayerNorm(normalized_shape, bias=False):
    return WithBiasLayerNorm(normalized_shape) if bias else BiasFreeLayerNorm(normalized_shape)

class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
        return self.project_out((attn @ v).reshape(b, c, h, w))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, volterra_rank):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=bias)
        self.attn = MDTA(dim, num_heads, bias)
        self.v1 = VolterraLayer2D(dim, dim, rank=volterra_rank)
        self.norm2 = LayerNorm(dim, bias=bias)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)
        self.v2 = VolterraLayer2D(dim, dim, rank=volterra_rank)
    def forward(self, x):
        x = x + self.v1(self.attn(self.norm1(x)))
        x = x + self.v2(self.ffn(self.norm2(x)))
        return x


# -----------------------------------------------
# Encoder / Decoder / Downsample / Upsample
# -----------------------------------------------
class Encoder(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.body = nn.Sequential(*[TransformerBlock(dim, **kwargs) for _ in range(depth)])
    def forward(self, x): return self.body(x)

class Decoder(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.body = nn.Sequential(*[TransformerBlock(dim, **kwargs) for _ in range(depth)])
    def forward(self, x): return self.body(x)

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(in_channels, in_channels * 2, 3, 2, 1)
    def forward(self, x): return self.body(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1),
            nn.PixelShuffle(2)
        )
    def forward(self, x): return self.body(x)


# -----------------------------------------------
# Final VETNet (Restormer + Volterra + FiLM)
# -----------------------------------------------
class VETNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, dim=64,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias',
        volterra_rank=4
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, 3, padding=1)

        self.encoder1 = Encoder(dim, num_blocks[0],
                                num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.film1 = FiLMBlock()
        self.down1 = Downsample(dim)

        self.encoder2 = Encoder(dim * 2, num_blocks[1],
                                num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.film2 = FiLMBlock()
        self.down2 = Downsample(dim * 2)

        self.encoder3 = Encoder(dim * 4, num_blocks[2],
                                num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.film3 = FiLMBlock()
        self.down3 = Downsample(dim * 4)

        self.latent = Encoder(dim * 8, num_blocks[3],
                              num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.film4 = FiLMBlock()

        self.up3 = Upsample(dim * 8, dim * 4)
        self.decoder3 = Decoder(dim * 4, num_blocks[2],
                                num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.up2 = Upsample(dim * 4, dim * 2)
        self.decoder2 = Decoder(dim * 2, num_blocks[1],
                                num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.up1 = Upsample(dim * 2, dim)
        self.decoder1 = Decoder(dim, num_blocks[0],
                                num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.refinement = Encoder(dim, num_refinement_blocks,
                                  num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.output = nn.Conv2d(dim, out_channels, 3, padding=1)

    def _pad_and_add(self, up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return up + skip

    def forward(self, x, film_signals=None):
        x1 = self.patch_embed(x)

        x2 = self.encoder1(x1)
        if film_signals: x2 = self.film1(x2, film_signals[0])

        x3 = self.encoder2(self.down1(x2))
        if film_signals: x3 = self.film2(x3, film_signals[1])

        x4 = self.encoder3(self.down2(x3))
        if film_signals: x4 = self.film3(x4, film_signals[2])

        x5 = self.latent(self.down3(x4))
        if film_signals: x5 = self.film4(x5, film_signals[3])

        d3 = self.decoder3(self._pad_and_add(self.up3(x5), x4))
        d2 = self.decoder2(self._pad_and_add(self.up2(d3), x3))
        d1 = self.decoder1(self._pad_and_add(self.up1(d2), x2))

        r = self.refinement(d1)
        out = self.output(r + x1)

        return torch.clamp(x + out, 0, 1)
