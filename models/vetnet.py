# 손 (Restoration Network)
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.film_layer import FiLMBlock

# ==============================================================================
# 🧩 Component 1: Volterra Layer (Non-linear Filtering)
# 역할: 2차 Volterra Series를 이용해 이미지의 비선형적 특징을 효과적으로 포착
# ==============================================================================

def circular_shift(x, shift_x, shift_y):
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))

class VolterraLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=2, use_lossless=False):
        super().__init__()
        self.use_lossless = use_lossless
        self.rank = rank

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

        if use_lossless:
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            self.shifts = self._generate_shifts(kernel_size)
        else:
            self.W2a = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                for _ in range(rank)
            ])
            self.W2b = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                for _ in range(rank)
            ])

    def _generate_shifts(self, k):
        P = k // 2
        shifts = []
        for s1 in range(-P, P + 1):
            for s2 in range(-P, P + 1):
                if s1 == 0 and s2 == 0: continue
                if (s1, s2) < (0, 0): continue
                shifts.append((s1, s2))
        return shifts

    def forward(self, x):
        linear_term = self.conv1(x)
        quadratic_term = 0

        if self.use_lossless:
            for s1, s2 in self.shifts:
                x_shifted = circular_shift(x, s1, s2)
                prod = x * x_shifted
                prod = torch.clamp(prod, min=-1.0, max=1.0)
                quadratic_term += self.conv2(prod)
        else:
            for a, b in zip(self.W2a, self.W2b):
                qa = torch.clamp(a(x), min=-1.0, max=1.0)
                qb = torch.clamp(b(x), min=-1.0, max=1.0)
                quadratic_term += qa * qb

        out = linear_term + quadratic_term
        return out

# ==============================================================================
# 🧱 Component 2: Restormer Building Blocks (GDFN, MDTA, TransformerBlock)
# ==============================================================================

class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        x = x / torch.sqrt(var + 1e-5)
        return self.weight * x

class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return self.weight * x + self.bias

def LayerNorm(normalized_shape, bias=False):
    return WithBiasLayerNorm(normalized_shape) if bias else BiasFreeLayerNorm(normalized_shape)

class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, volterra_rank):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=bias) if LayerNorm_type == 'WithBias' else BiasFreeLayerNorm(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.volterra1 = VolterraLayer2D(dim, dim, rank=volterra_rank)

        self.norm2 = LayerNorm(dim, bias=bias) if LayerNorm_type == 'WithBias' else BiasFreeLayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)
        self.volterra2 = VolterraLayer2D(dim, dim, rank=volterra_rank)

    def forward(self, x):
        # Attention + Volterra
        x = x + self.volterra1(self.attn(self.norm1(x)))
        # FFN + Volterra
        x = x + self.volterra2(self.ffn(self.norm2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.body = nn.Sequential(*[TransformerBlock(dim, **kwargs) for _ in range(depth)])
    def forward(self, x):
        return self.body(x)

class Decoder(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.body = nn.Sequential(*[TransformerBlock(dim, **kwargs) for _ in range(depth)])
    def forward(self, x):
        return self.body(x)

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2) 
        )
    def forward(self, x):
        return self.body(x)

# ==============================================================================
# 🖐️ The Hands: VETNet (Restormer + Volterra + FiLM Controller)
# 역할: 고해상도 이미지 복원 (Denoising, Deblurring 등)
# 통합: 사용자 정의 RestormerVolterra 구조에 LLM 제어 신호(FiLM)를 주입
# ==============================================================================

class VETNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64, num_blocks=[4,6,6,8],
                 num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias', volterra_rank=4):
        """
        Args:
            dim: 기본 채널 수 (Default 64 -> Levels: 64, 128, 256, 512)
                 (config.yaml의 vetnet_channels와 일치시킴)
        """
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        # Level 1
        self.encoder1 = Encoder(dim, num_blocks[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.film1 = FiLMBlock() # ⚡ FiLM Injection Point 1 (Channel: dim)
        self.down1 = Downsample(dim)

        # Level 2
        self.encoder2 = Encoder(dim*2, num_blocks[1], num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.film2 = FiLMBlock() # ⚡ FiLM Injection Point 2 (Channel: dim*2)
        self.down2 = Downsample(dim*2)

        # Level 3
        self.encoder3 = Encoder(dim*4, num_blocks[2], num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.film3 = FiLMBlock() # ⚡ FiLM Injection Point 3 (Channel: dim*4)
        self.down3 = Downsample(dim*4)

        # Level 4 (Latent)
        self.latent = Encoder(dim*8, num_blocks[3], num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        # Decoder Path
        self.up3 = Upsample(dim*8, dim*4)
        self.decoder3 = Decoder(dim*4, num_blocks[2], num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.up2 = Upsample(dim*4, dim*2)
        self.decoder2 = Decoder(dim*2, num_blocks[1], num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.up1 = Upsample(dim*2, dim)
        self.decoder1 = Decoder(dim, num_blocks[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.refinement = Encoder(dim, num_refinement_blocks, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def _pad_and_add(self, up_tensor, skip_tensor):
        if up_tensor.shape[-2:] != skip_tensor.shape[-2:]:
            up_tensor = F.interpolate(up_tensor, size=skip_tensor.shape[-2:], mode='bilinear', align_corners=False)
        return up_tensor + skip_tensor

    def forward(self, x, film_signals=None):
        """
        Args:
            x: Input Image [B, 3, H, W]
            film_signals: List of tuples [(gamma, beta), ...]. 
                          Expected Order: [Level1, Level2, Level3, ...]
        """
        # Input Embedding
        x1 = self.patch_embed(x)
        
        # Encoder 1
        x2 = self.encoder1(x1)
        if film_signals is not None:
             x2 = self.film1(x2, film_signals[0]) # ⚡ Apply FiLM (64ch)
        
        # Encoder 2
        x3 = self.encoder2(self.down1(x2))
        if film_signals is not None:
             x3 = self.film2(x3, film_signals[1]) # ⚡ Apply FiLM (128ch)

        # Encoder 3
        x4 = self.encoder3(self.down2(x3))
        if film_signals is not None:
             x4 = self.film3(x4, film_signals[2]) # ⚡ Apply FiLM (256ch)

        # Latent
        x5 = self.latent(self.down3(x4))
        
        # Decoder
        x6 = self.decoder3(self._pad_and_add(self.up3(x5), x4))
        x7 = self.decoder2(self._pad_and_add(self.up2(x6), x3))
        x8 = self.decoder1(self._pad_and_add(self.up1(x7), x2))
        
        # Refinement
        x9 = self.refinement(x8)
        
        # Output (Residual Learning)
        out = self.output(x9 + x1)
        return x + out