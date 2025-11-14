# G:\DPR-Net\model\dpr_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# 'torchvision'은 CLIP 정규화를 위해 필요합니다. (pip install torchvision)
import torchvision.transforms as T 

# --- 1. 모듈 임포트 ---

# VETNet (Hands) 백본 및 구성요소
# (RestormerVolterra가 Encoder, Decoder 등을 export하지 않으므로, 
#  상속(inheritance)을 사용해 forward pass만 수정합니다.)
from .restormer_volterra import RestormerVolterra

# DPR-Net V2 모듈 (Eyes, Brain, Controller)
from .dpr_modules import DprClipEncoder, DprLLM, DprMultiStageFiLMHead
# 테스트용 더미 LLM 모듈
from .dpr_modules import DummyDprLLM 


# -----------------------------------------------
#  ⑤ FiLM이 주입되는 VETNet (Hands)
# -----------------------------------------------

class FiLMedVETNet(RestormerVolterra):
    """
    DPR-Net의 FiLM 신호를 주입받도록
    기존 RestormerVolterra(VETNet)의 forward pass를
    수정한 LLM 제어형 백본 모듈입니다.
    
    (부모 클래스의 모든 init 로직을 상속받습니다.)
    """
    def __init__(self, *args, **kwargs):
        # 부모 클래스(RestormerVolterra)의 __init__을 그대로 호출
        super().__init__(*args, **kwargs)
        print("FiLMedVETNet (Hands): Initialized (inherits from RestormerVolterra).")

    def _apply_film(self, x, film_signal):
        """ FiLM 신호 (gamma, beta)를 피처맵(x)에 적용 """
        if film_signal is None:
            # 혹시 모를 상황 대비 (신호가 없으면 x 그대로 반환)
            return x
        
        gamma, beta = film_signal
        
        # V2 아키텍처: F_i_out = F_i * γ_i + β_i
        # (Restormer의 skip-connection과 유사하게, 1 + gamma를 사용하면 더 안정적일 수 있음)
        # return x * (1 + gamma) + beta 
        return x * gamma + beta

    # *** VETNet(RestormerVolterra)의 Forward Pass 오버라이드 ***
    def forward(self, x, film_signals: dict):
        """
        Args:
            x (torch.Tensor): 입력 이미지 (B, C, H, W)
            film_signals (dict): DprMultiStageFiLMHead에서 생성된
                                 스테이지별 (gamma, beta) 튜플 딕셔너리.
        """
        
        # --- Encoder Path (FiLM 주입) ---
        x1 = self.patch_embed(x)
        
        # Encoder 1
        x2 = self.encoder1(x1)
        x2 = self._apply_film(x2, film_signals.get('encoder1'))

        # Encoder 2
        x3 = self.encoder2(self.down1(x2))
        x3 = self._apply_film(x3, film_signals.get('encoder2'))

        # Encoder 3
        x4 = self.encoder3(self.down2(x3))
        x4 = self._apply_film(x4, film_signals.get('encoder3'))
        
        # Latent
        x5 = self.latent(self.down3(x4))
        x5 = self._apply_film(x5, film_signals.get('latent'))

        # --- Decoder Path (FiLM 주입) ---
        
        # Decoder 3
        x6 = self.decoder3(self._pad_and_add(self.up3(x5), x4))
        x6 = self._apply_film(x6, film_signals.get('decoder3'))
        
        # Decoder 2
        x7 = self.decoder2(self._pad_and_add(self.up2(x6), x3))
        x7 = self._apply_film(x7, film_signals.get('decoder2'))
        
        # Decoder 1
        x8 = self.decoder1(self._pad_and_add(self.up1(x7), x2))
        x8 = self._apply_film(x8, film_signals.get('decoder1'))
        
        # Refinement
        x9 = self.refinement(x8)
        x9 = self._apply_film(x9, film_signals.get('refinement'))
        
        # Final output
        # (원본 VETNet과 동일하게 residual connection 적용)
        out = self.output(x9 + x1)
        
        return out


# -----------------------------------------------
#  🚀 DPR-Net (V2) 최종 조립 모델
# -----------------------------------------------

class DPR_Net(nn.Module):
    """
    DPR-Net (V2) - Dual-Path Restoration Network
    모든 모듈(Eyes, Brain, Controller, Hands)을 조립한
    최종 엔드-투-엔드 모델입니다.
    """
    def __init__(self, 
                 # VETNet Backbone Config (restormer_volterra.py 기본값)
                 vetnet_dim=48, 
                 vetnet_num_blocks=[4,6,6,8],
                 vetnet_refinement_blocks=4,
                 vetnet_heads=[1,2,4,8],
                 vetnet_ffn_exp=2.66,
                 vetnet_bias=False,
                 vetnet_ln_type='WithBias',
                 vetnet_volterra_rank=4,
                 
                 # CLIP Config (dpr_modules.py 기본값)
                 clip_model_name="ViT-L-14",
                 clip_pretrained="laion2b_s32b_b82k",
                 clip_embed_dim=1024, # ViT-L-14 default
                 
                 # LLM Config (dpr_modules.py 기본값)
                 llm_model_name="mistralai/Mistral-7B-v0.1",
                 llm_embed_dim=4096 # Mistral-7B default
                ):
        super().__init__()
        
        print("Initializing DPR-Net (V2)...")
        
        # --- 1. Eyes (CLIP) ---
        print("  (1/4) Loading Eyes (CLIP)...")
        self.clip_encoder = DprClipEncoder(
            model_name=clip_model_name,
            pretrained=clip_pretrained
        )
        # CLIP이 요구하는 224x224 정규화 (모델 내부에 포함)
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        self.clip_normalize = T.Normalize(mean=clip_mean, std=clip_std)
        
        # --- 2. Brain (LLM) ---
        print("  (2/4) Loading Brain (LLM)...")
        self.llm_module = DprLLM(
            clip_embed_dim=clip_embed_dim,
            llm_embed_dim=llm_embed_dim,
            model_name=llm_model_name
        )
        self.tokenizer = self.llm_module.tokenizer # for text generation

        # --- 3. Controller (FiLM Head) ---
        print("  (3/4) Loading Controller (FiLM Head)...")
        self.film_head = DprMultiStageFiLMHead(
            llm_embed_dim=llm_embed_dim,
            vetnet_dim=vetnet_dim
        )
        
        # --- 4. Hands (FiLM-Controlled VETNet) ---
        print("  (4/4) Loading Hands (FiLMed VETNet Backbone)...")
        self.vet_backbone = FiLMedVETNet(
            dim=vetnet_dim,
            num_blocks=vetnet_num_blocks,
            num_refinement_blocks=vetnet_refinement_blocks,
            heads=vetnet_heads,
            ffn_expansion_factor=vetnet_ffn_exp,
            bias=vetnet_bias,
            LayerNorm_type=vetnet_ln_type,
            volterra_rank=vetnet_volterra_rank
        )
        
        print("\n✅ DPR-Net (V2) successfully initialized.")

    def forward(self, img_distorted):
        """
        DPR-Net의 V2 듀얼-패스 forward를 수행합니다.
        
        Args:
            img_distorted (torch.Tensor): (B, 3, H, W) 
                                        원본 해상도의 0~1 범위 텐서.
        
        Returns:
            dict: {
                'img_restored': (B, 3, H, W) - 복원된 이미지,
                'logits': (B, N+1, Vocab) - Text Path V2 (L_consistency, Text gen용),
                'hidden_state': (B, N+1, D_llm) - Control Path V2 (L_consistency용),
                'film_signals': (dict) - VETNet에 주입된 (gamma, beta) 딕셔너리
            }
        """
        
        # --- (A) Control/Text Path (LLM) ---
        
        # 1. Preprocess for CLIP (Resize to 224x224, Normalize)
        img_clip = F.interpolate(img_distorted, 
                                 size=224, 
                                 mode='bicubic', 
                                 align_corners=False)
        img_clip_norm = self.clip_normalize(img_clip)
        
        # 2. Eyes -> Brain -> Controller
        vision_tokens, _ = self.clip_encoder(img_clip_norm)
        hidden_state, logits = self.llm_module(vision_tokens)
        film_signals = self.film_head(hidden_state) 
        
        # --- (B) Restoration Path (VETNet) ---
        
        # 3. Hands (VETNet)
        # 원본 해상도 이미지(img_distorted)와 FiLM 신호(film_signals)를 VETNet에 주입
        img_restored = self.vet_backbone(img_distorted, film_signals)
        
        # --- (C) Outputs ---
        
        # V2 훈련(L_total) 및 추론(Image+Text)에 필요한 모든 출력을 반환
        return {
            'img_restored': img_restored,  # L_img
            'logits': logits,              # L_consistency, L_text
            'hidden_state': hidden_state,  # L_consistency
            'film_signals': film_signals   # L_film (stage alignment)
        }


# -----------------------------------------------
#  ✅ 테스트 코드 (dpr_net.py)
# -----------------------------------------------

class DummyDPR_Net(DPR_Net):
    """
    Mistral-7B 다운로드 없이 빠른 테스트를 위한
    DPR_Net의 더미 버전입니다.
    """
    def __init__(self, *args, **kwargs):
        # `DPR_Net`의 __init__을 호출하되,
        # `DprLLM` 로딩 부분을 `DummyDprLLM`으로 덮어씁니다.
        
        # 1. VETNet/CLIP/FiLMHead config
        vetnet_dim = kwargs.get('vetnet_dim', 48)
        clip_embed_dim = kwargs.get('clip_embed_dim', 1024)
        llm_embed_dim = kwargs.get('llm_embed_dim', 4096)
        
        # `nn.Module.__init__`을 먼저 호출
        nn.Module.__init__(self) 
        
        print("[Dummy Mode] Initializing DPR-Net (V2)...")
        
        # --- 1. Eyes (CLIP) ---
        print("  (1/4) [Dummy Mode] Loading Eyes (CLIP)...")
        self.clip_encoder = DprClipEncoder(
            model_name=kwargs.get('clip_model_name', "ViT-L-14"),
            pretrained=kwargs.get('clip_pretrained', "laion2b_s32b_b82k")
        )
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        self.clip_normalize = T.Normalize(mean=clip_mean, std=clip_std)
        
        # --- 2. Brain (LLM) ---
        print("  (2/4) [Dummy Mode] Loading Brain (Dummy LLM)...")
        self.llm_module = DummyDprLLM( # *** 여기가 다름 ***
            clip_embed_dim=clip_embed_dim,
            llm_embed_dim=llm_embed_dim
        )
        self.tokenizer = self.llm_module.tokenizer

        # --- 3. Controller (FiLM Head) ---
        print("  (3/4) [Dummy Mode] Loading Controller (FiLM Head)...")
        self.film_head = DprMultiStageFiLMHead(
            llm_embed_dim=llm_embed_dim,
            vetnet_dim=vetnet_dim
        )
        
        # --- 4. Hands (FiLM-Controlled VETNet) ---
        print("  (4/4) [Dummy Mode] Loading Hands (FiLMed VETNet Backbone)...")
        self.vet_backbone = FiLMedVETNet(
            dim=vetnet_dim,
            num_blocks=kwargs.get('vetnet_num_blocks', [4,6,6,8]),
            num_refinement_blocks=kwargs.get('vetnet_refinement_blocks', 4),
            heads=kwargs.get('vetnet_heads', [1,2,4,8]),
            ffn_expansion_factor=kwargs.get('vetnet_ffn_exp', 2.66),
            bias=kwargs.get('vetnet_bias', False),
            LayerNorm_type=kwargs.get('vetnet_ln_type', 'WithBias'),
            volterra_rank=kwargs.get('vetnet_volterra_rank', 4)
        )
        
        print("\n✅ [Dummy Mode] DPR-Net (V2) successfully initialized.")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (ViT-L-14 @ 1024 dim -> Mistral @ 4096 dim)
    # (테스트를 위해 백본 크기를 줄임)
    config = {
        'vetnet_dim': 48,
        'vetnet_num_blocks': [2, 2, 2, 2], # Smaller for fast test
        'vetnet_refinement_blocks': 2,
        'vetnet_heads': [1,2,4,8],
        'clip_embed_dim': 1024, # ViT-L-14
        'llm_embed_dim': 4096,  # Mistral-7B
    }
    
    # 원본 해상도 (홀수, 비대칭) - VETNet 강인성 테스트
    B, C, H, W = 2, 3, 127, 133
    # 0~1 범위의 랜덤 이미지 (일반적인 데이터셋 입력)
    dummy_image = torch.rand(B, C, H, W).to(device)
    
    print(f"--- DPR-Net (V2) End-to-End Test (on {device}) ---")
    
    try:
        # `DummyDPR_Net`을 사용하여 모델 다운로드 방지
        model = DummyDPR_Net(**config).to(device)
        model.eval()
        
        print(f"\nInput image shape: {dummy_image.shape}")
        
        with torch.no_grad():
            outputs = model(dummy_image)
        
        print("\n--- Model Output Verification ---")
        print(f"Type of output: {type(outputs)}")
        assert isinstance(outputs, dict)
        
        # 1. img_restored (복원 이미지)
        print(f"Restored image shape: {outputs['img_restored'].shape}")
        assert outputs['img_restored'].shape == (B, C, H, W) # 원본 해상도 유지
        
        # 2. logits (텍스트 경로)
        # (B, N+1, Vocab) - N+1 = 257 (ViT-L-14 @ 224px)
        num_tokens = 257 
        vocab_size = model.llm_module.config.vocab_size
        print(f"Logits shape: {outputs['logits'].shape}")
        assert outputs['logits'].shape == (B, num_tokens, vocab_size)
        
        # 3. hidden_state (제어 경로)
        # (B, N+1, D_llm)
        llm_dim = config['llm_embed_dim']
        print(f"Hidden state shape: {outputs['hidden_state'].shape}")
        assert outputs['hidden_state'].shape == (B, num_tokens, llm_dim)
        
        # 4. film_signals (제어 신호)
        num_stages = model.film_head.num_stages # 8개
        print(f"FiLM signals dict size: {len(outputs['film_signals'])}")
        assert len(outputs['film_signals']) == num_stages
        
        print("\n✅ DPR-Net (V2) End-to-End test passed!")
        
    except ImportError as e:
        print(f"\n[Error] 필요한 모듈을 G:\\DPR-Net\\model\\ 폴더에서 찾을 수 없습니다.")
        print(f"Make sure '__init__.py', 'dpr_modules.py', 'restormer_volterra.py' exist.")
        print(f"Details: {e}")
    except Exception as e:
        import traceback
        print(f"\n테스트 중 오류 발생: {e}")
        traceback.print_exc()