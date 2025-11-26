# 뇌 -> 공간 변환 (Pixel Decoder)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 🧩 The Brain Part 2: Optimized Pixel Decoder
# 역할: LLM의 추상적인 텍스트/이미지 혼합 사고를 다시 '공간적(Spatial)'인 맵으로 변환
# 핵심: Reshape -> Bottleneck -> PPM(Context) -> Restoration -> Fusion
# ==============================================================================

class PixelDecoder(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, output_dim=4096):
        """
        Args:
            input_dim: Mistral Hidden Size (4096)
            hidden_dim: Bottleneck Channel (512) - 연산량 감소용
            output_dim: Final Feature Size (4096) - 다시 Mistral 차원으로 복구
        """
        super().__init__()
        
        # 1. Bottleneck (Down-projection)
        # [B, 4096, 16, 16] -> [B, 512, 16, 16]
        # 채널을 1/8로 줄여 연산 효율성을 확보하고 특징을 압축합니다.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, hidden_dim), # 학습 안정성을 위해 GroupNorm 사용
            nn.GELU()
        )
        
        # 2. Refined PPM (Pyramid Pooling Module)
        # 다양한 스케일(1x1, 2x2, 4x4, 8x8)로 이미지를 봐서 문맥 정보를 수집
        self.ppm_scales = [1, 2, 4, 8]
        self.ppm_pooling = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in self.ppm_scales
        ])
        
        # 각 스케일별 특징을 처리할 작은 Conv
        self.ppm_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1),
                nn.GELU()
            ) for _ in range(len(self.ppm_scales))
        ])
        
        # 3. Spatial Conv (Feature Aggregation)
        # Original Bottleneck (512) + 4 PPM Scales (128 * 4 = 512) = 1024 Channels
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(hidden_dim + (hidden_dim // 4) * len(self.ppm_scales), hidden_dim, 
                      kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU()
        )
        
        # 4. Restoration (Up-projection)
        # [B, 512, 16, 16] -> [B, 4096, 16, 16]
        self.restore = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        
        # 5. Final Normalization
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, llm_hidden_state):
        """
        Args:
            llm_hidden_state: [Batch, 257 + Text_Len, 4096] (Mistral의 전체 출력)
            
        Returns:
            vision_features: [Batch, 257, 4096] (공간 정보가 복원된 Vision 토큰들)
        """
        # ----------------------------------------------------------------------
        # Step 1: Token Slicing (Vision Part Extraction)
        # 명세서: "뒤에 붙은 텍스트 토큰은 버리고, 앞쪽 257개만 잘라냅니다."
        # ----------------------------------------------------------------------
        vision_tokens = llm_hidden_state[:, :257, :] # [B, 257, 4096]
        
        # CLS(전역 정보)와 Patch(지역 정보) 분리
        cls_token = vision_tokens[:, 0:1, :]   # [B, 1, 4096]
        patch_tokens = vision_tokens[:, 1:, :] # [B, 256, 4096]
        
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5) # 16 (16x16 = 256)
        
        # ----------------------------------------------------------------------
        # Step 2: Reshape to Spatial Map
        # 1D Sequence -> 2D Image Map 변환
        # [B, 256, 4096] -> [B, 4096, 256] -> [B, 4096, 16, 16]
        # ----------------------------------------------------------------------
        x = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        # ----------------------------------------------------------------------
        # Step 3: Bottleneck & PPM (Context Enhancement)
        # ----------------------------------------------------------------------
        x_bn = self.bottleneck(x) # [B, 512, 16, 16]
        
        ppm_outs = [x_bn]
        for pool, conv in zip(self.ppm_pooling, self.ppm_convs):
            feat = pool(x_bn)
            feat = conv(feat)
            # ⚠️ Key Update: Bilinear Interpolation (align_corners=False)
            # 작은 맵을 다시 16x16으로 키울 때 격자 무늬 방지
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            ppm_outs.append(feat)
            
        # Concatenation: 512 + (128*4) = 1024 channels
        x_ppm = torch.cat(ppm_outs, dim=1) 
        
        # ----------------------------------------------------------------------
        # Step 4: Spatial Convolution & Restoration
        # ----------------------------------------------------------------------
        x_spatial = self.spatial_conv(x_ppm) # [B, 512, 16, 16]
        x_restored = self.restore(x_spatial) # [B, 4096, 16, 16]
        
        # ----------------------------------------------------------------------
        # Step 5: Flatten & Feature Fusion
        # 다시 토큰 형태로 변환: [B, 4096, 16, 16] -> [B, 256, 4096]
        # ----------------------------------------------------------------------
        x_flat = x_restored.flatten(2).transpose(1, 2)
        
        # 1. Residual Connection (Skip Connection)
        # 원래의 patch_tokens에 처리된 정보(x_flat)를 더함 -> 학습 안정성
        refined_patches = patch_tokens + x_flat
        
        # 2. Global Context Fusion (Broadcasting)
        # CLS 토큰(이미지 전체 요약)을 모든 패치 픽셀에 더해줌
        refined_patches = refined_patches + cls_token
        
        # ----------------------------------------------------------------------
        # Step 6: Final Assembly
        # CLS 토큰과 정제된 패치를 다시 합침
        # ----------------------------------------------------------------------
        out = torch.cat([cls_token, refined_patches], dim=1) # [B, 257, 4096]
        
        # 최종 정규화
        out = self.final_norm(out)
        
        return out