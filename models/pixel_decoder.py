# 뇌 -> 공간 변환 (Pixel Decoder)
# G:\DPR-Net\models\pixel_decoder.py
# G:\DPR-Net\models\pixel_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelDecoder(nn.Module):
    """
    DPR-Net V2 - PixelLM Spatial Decoder
    역할: [B, 257 + T, 4096] → [B, 257, 4096]
    """
    def __init__(self, input_dim=4096, hidden_dim=512, output_dim=4096):
        super().__init__()

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU()
        )

        # PPM branches
        self.ppm_scales = [1, 2, 4, 8]
        self.ppm_pooling = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in self.ppm_scales
        ])
        self.ppm_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1, bias=False),
                nn.GELU()
            ) for _ in self.ppm_scales
        ])

        # [512 + 128×4] → 512
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(hidden_dim + (hidden_dim // 4) * len(self.ppm_scales),
                      hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU()
        )

        # Restore dimension: 512 → 4096
        self.restore = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

        # Learnable fusion scale for CLS (훈련 안정성 핵심)
        self.alpha = nn.Parameter(torch.ones(1) * 1e-4)

        # Norm
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, llm_hidden_state):
        """
        llm_hidden_state: [B, 257 + T, 4096]
        return: [B, 257, 4096]
        """
        # ---- Token Slice ----
        vision_tokens = llm_hidden_state[:, :257, :]  # (B, 257, 4096)
        cls_token     = vision_tokens[:, :1, :]       # (B, 1, 4096)
        patch_tokens  = vision_tokens[:, 1:, :]       # (B, 256, 4096)

        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)  # 16

        # ---- 1D → 2D ----
        x = patch_tokens.transpose(1, 2).reshape(B, C, H, W)

        # ---- Bottleneck ----
        x = self.bottleneck(x)  # (B, 512, 16,16)

        # ---- PPM ----
        ppm_outs = [x]
        for pool, conv in zip(self.ppm_pooling, self.ppm_convs):
            y = conv(pool(x))
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            ppm_outs.append(y)
        x = torch.cat(ppm_outs, dim=1)  # (B, 1024, 16,16)

        # ---- Spatial Conv + Restore ----
        x = self.spatial_conv(x)   # (B, 512, 16,16)
        x = self.restore(x)        # (B, 4096, 16,16)

        # ---- Flatten ----
        x = x.flatten(2).transpose(1, 2)  # (B, 256, 4096)

        # ---- Residual Fusion + Soft Global Fusion (안정화 핵심) ----
        refined = patch_tokens + x + self.alpha * cls_token  # no broadcast explosion

        # ---- Assemble CLS + patches ----
        out = torch.cat([cls_token, refined], dim=1)  # (B, 257, 4096)

        # ---- Final Norm ----
        return self.final_norm(out)
