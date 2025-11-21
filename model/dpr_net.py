
# G:\DPR-Net\model\dpr_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T 

# --- 1. 모듈 임포트 ---
# [수정] DprLLM 대신 DprPixelLM을 가져옵니다.
from .restormer_volterra import RestormerVolterra
from .dpr_modules import DprClipEncoder, DprPixelLM, DprMultiStageFiLMHead

# -----------------------------------------------
#  ⑤ FiLM이 주입되는 VETNet (Hands)
# -----------------------------------------------
class FiLMedVETNet(RestormerVolterra):
    # (이 클래스는 변경 사항 없음)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("FiLMedVETNet (Hands): Initialized (inherits from RestormerVolterra).")

    def _apply_film(self, x, film_signal):
        if film_signal is None:
            return x
        gamma, beta = film_signal
        return x * gamma + beta

    def forward(self, x, film_signals: dict):
        x1 = self.patch_embed(x)
        x2 = self.encoder1(x1)
        x2 = self._apply_film(x2, film_signals.get('encoder1'))
        x3 = self.encoder2(self.down1(x2))
        x3 = self._apply_film(x3, film_signals.get('encoder2'))
        x4 = self.encoder3(self.down2(x3))
        x4 = self._apply_film(x4, film_signals.get('encoder3'))
        x5 = self.latent(self.down3(x4))
        x5 = self._apply_film(x5, film_signals.get('latent'))
        x6 = self.decoder3(self._pad_and_add(self.up3(x5), x4))
        x6 = self._apply_film(x6, film_signals.get('decoder3'))
        x7 = self.decoder2(self._pad_and_add(self.up2(x6), x3))
        x7 = self._apply_film(x7, film_signals.get('decoder2'))
        x8 = self.decoder1(self._pad_and_add(self.up1(x7), x2))
        x8 = self._apply_film(x8, film_signals.get('decoder1'))
        x9 = self.refinement(x8)
        x9 = self._apply_film(x9, film_signals.get('refinement'))
        out = self.output(x9 + x1)
        return out

# -----------------------------------------------
#  🚀 DPR-Net (V2) Final Assembly [PixelLM Powered]
# -----------------------------------------------
class DPR_Net(nn.Module):
    def __init__(self, 
                 vetnet_dim=48, vetnet_num_blocks=[4,6,6,8],
                 vetnet_refinement_blocks=4, vetnet_heads=[1,2,4,8],
                 vetnet_ffn_exp=2.66, vetnet_bias=False,
                 vetnet_ln_type='WithBias', vetnet_volterra_rank=4,
                 clip_model_name="ViT-L-14", clip_pretrained="laion2b_s32b_b82k",
                 clip_embed_dim=1024, 
                 llm_model_name="mistralai/Mistral-7B-v0.1",
                 llm_embed_dim=4096 
                ):
        super().__init__()
        
        print("Initializing DPR-Net (V2) [PixelLM Powered]...")
        
        # --- 1. Eyes (CLIP) ---
        print(" (1/4) Loading Eyes (CLIP)...")
        self.clip_encoder = DprClipEncoder(
            model_name=clip_model_name, pretrained=clip_pretrained
        )
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        self.clip_normalize = T.Normalize(mean=clip_mean, std=clip_std)
        
        # --- 2. Brain (PixelLM) ---
        # [수정] DprPixelLM으로 교체
        print(" (2/4) Loading Brain (PixelLM with Decoder)...")
        self.llm_module = DprPixelLM(
            clip_embed_dim=clip_embed_dim,
            llm_embed_dim=llm_embed_dim,
            model_name=llm_model_name
        )
        self.tokenizer = self.llm_module.tokenizer

        # --- 3. Controller (FiLM Head) ---
        print(" (3/4) Loading Controller (FiLM Head)...")
        self.film_head = DprMultiStageFiLMHead(
            llm_embed_dim=llm_embed_dim,
            vetnet_dim=vetnet_dim
        )
        
        # --- 4. Hands (FiLM-Controlled VETNet) ---
        print(" (4/4) Loading Hands (FiLMed VETNet Backbone)...")
        self.vet_backbone = FiLMedVETNet(
            dim=vetnet_dim, num_blocks=vetnet_num_blocks,
            num_refinement_blocks=vetnet_refinement_blocks,
            heads=vetnet_heads, ffn_expansion_factor=vetnet_ffn_exp,
            bias=vetnet_bias, LayerNorm_type=vetnet_ln_type,
            volterra_rank=vetnet_volterra_rank
        )
        
        print("\n✅ DPR-Net (V2) [PixelLM Powered] successfully initialized.")

    def forward(self, img_distorted, 
                text_labels=None, 
                text_attention_mask=None, 
                mode='train'):
        
        # 1. Preprocess for CLIP
        img_clip = F.interpolate(img_distorted, size=224, mode='bicubic', align_corners=False)
        img_clip_norm = self.clip_normalize(img_clip)
        
        # 2. Eyes -> Brain
        vision_tokens, _ = self.clip_encoder(img_clip_norm)
        
        if mode == 'train':
            # 3. 훈련 모드 (PixelLM Forward)
            # 반환값: loss_text, pixel_decoded_features, logits
            loss_text, hidden_state, logits = self.llm_module(
                vision_tokens, 
                text_labels=text_labels, 
                text_attention_mask=text_attention_mask
            )
            
            # 4. FiLM 신호 생성
            # hidden_state는 이미 Pixel Decoder를 통과하여 (B, 257, D) 형태입니다.
            # Vision 부분(Decoder 통과됨)만 잘라낼 필요 없이 그대로 사용하면 되지만,
            # 기존 코드 로직에 따라 Vision Tokens 길이만큼 슬라이싱 (257개)
            vision_hidden_state = hidden_state[:, :vision_tokens.shape[1], :]
            
            film_signals = self.film_head(vision_hidden_state) 
            cls_hidden_state = vision_hidden_state[:, 0:1, :] 
            
            # 5. VETNet 실행
            img_restored = self.vet_backbone(img_distorted, film_signals)
            
            # 6. 훈련용 출력 반환
            return {
                'img_restored': img_restored,
                'loss_text': loss_text,
                'cls_hidden_state': cls_hidden_state,
                'logits': logits,
            }

        else: # (mode == 'eval')
            # 3. 추론 모드
            generated_ids, hidden_state, _ = self.llm_module(vision_tokens) 
            
            # 4. FiLM 신호 생성
            # Pixel Decoder를 통과한 Feature를 사용
            film_signals = self.film_head(hidden_state[:, :vision_tokens.shape[1], :]) 
            
            # 5. VETNet 실행
            img_restored = self.vet_backbone(img_distorted, film_signals)

            # 6. 추론용 출력 반환
            return {
                'img_restored': img_restored,
                'generated_ids': generated_ids,
                'film_signals': film_signals 
            }


if __name__ == '__main__':
    # 간단한 의존성 및 초기화 테스트
    try:
        model = DPR_Net()
        print("DPR_Net instantiation successful.")
    except Exception as e:
        print(f"Initialization check failed (expected if no GPU/Weights): {e}")