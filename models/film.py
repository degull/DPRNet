# G:\DPR-Net\models\film.py
import torch
import torch.nn as nn

# ==============================================================================
# 🎛️ FiLM Generator (LLM → Restoration Controller)
#  - CLS + patch mean fusion (전역 + 지역 컨텍스트 결합)
#  - Zero-init + gamma scaling 안전장치
#  - VETNet Stage별 (gamma, beta) 제어 신호 생성
# ==============================================================================

class FiLMGenerator(nn.Module):
    def __init__(self, input_dim=4096, vetnet_channels=[64, 128, 256, 512]):
        """
        Args:
            input_dim        : PixelDecoder 출력 채널 (4096)
            vetnet_channels  : 각 VETNet stage의 채널 수 (예: [64, 128, 256, 512])
        """
        super().__init__()
        self.input_dim = input_dim
        self.channels = vetnet_channels

        # 🔥 CLS + mean patch token → Stronger context
        self.mix = nn.Linear(input_dim * 2, input_dim)

        # Stage별 FiLM 헤드 생성
        self.heads = nn.ModuleList()
        for ch in vetnet_channels:
            head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, ch * 2)  # gamma + beta
            )
            # Zero-init last layer → 학습 초기는 identity (안정한 시작)
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.heads.append(head)

        # 🔒 γ 폭주 방지: 크기 제한
        self.gamma_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, llm_features):
        """
        Args:
            llm_features : [B, 257, 4096] (CLS + 256 patches)
        Returns:
            film_signals : List of (gamma, beta)
        """
        B, N, C = llm_features.shape

        # Global + Local fusion
        cls = llm_features[:, 0, :]               # (B, 4096)
        patch_mean = llm_features[:, 1:, :].mean(dim=1)  # (B, 4096)
        mixed = torch.cat([cls, patch_mean], dim=1)       # (B, 8192)

        mixed = self.mix(mixed)  # (B, 4096)

        film_signals = []
        for head in self.heads:
            params = head(mixed)                # (B, 2ch)
            gamma, beta = params.chunk(2, dim=1)

            # 안정장치
            gamma = self.gamma_scale * gamma.tanh()

            # Conv2d 적용 형태로 reshape
            gamma = gamma[:, :, None, None]     # (B, ch, 1, 1)
            beta  = beta[:, :, None, None]      # (B, ch, 1, 1)

            film_signals.append((gamma, beta))

        return film_signals


# ==============================================================================
# 🔌 FiLM Block (VETNet 내부에 적용되는 연산)
#  x_new = x * (1 + gamma) + beta
#  (Residual scaling → 학습 안정성)
# ==============================================================================

class FiLMBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, film_signal):
        """
        Args:
            x           : Feature map [B, ch, H, W]
            film_signal : (gamma, beta) tuple
        """
        gamma, beta = film_signal
        x = x * (1 + gamma)  # scale
        x = x + beta         # shift
        return x
