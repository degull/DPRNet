# 🏗️ 전체 조립 (Main Model)
# G:\DPR-Net\models\dpr_net_v2.py
print("🔥 dpr_net_v2.py LOADED FROM:", __file__)

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

# 각 모듈 가져오기
from models.clip_encoder import CLIPVisionEncoder
from models.mistral_llm import MistralLLM
from models.pixel_decoder import PixelDecoder
from models.film import FiLMGenerator
from models.vetnet import VETNet

# ==============================================================================
# 🏗️ DPR-Net V2: The Final Assembly
# "From Auto-Captioning to Spatial Reconstruction"
#
# 구조:
# 1. Eyes (CLIP)        -> 이미지 특징 추출 (Frozen)
# 2. Brain (Mistral)    -> 텍스트+이미지 멀티모달 추론 (LoRA)
# 3. Translator         -> LLM 사고를 공간 맵으로 변환 (PixelDecoder)
# 4. Controller (FiLM)  -> 복원 제어 신호(gamma, beta) 생성
# 5. Hands (VETNet)     -> 고해상도 이미지 복원 수행
# ==============================================================================

class DPRNetV2(nn.Module):
    def __init__(self, config):
        """
        Args:
            config (dict or object): 모델 설정값 (dpr_config.yaml 내용)
        """
        super().__init__()

        # 설정값 로드 (Dictionary or Namespace 처리)
        if isinstance(config, dict):
            llm_id       = config["model"]["llm_model_id"]
            vision_id    = config["model"]["vision_model_id"]
            vet_channels = config["model"]["vetnet_channels"]
        else:
            llm_id       = config.model.llm_model_id
            vision_id    = config.model.vision_model_id
            vet_channels = config.model.vetnet_channels

        print("\n" + "=" * 60)
        print("🏗️ Initializing DPR-Net V2 (PixelLM Powered)...")
        print("=" * 60)

        # [2단계: 눈] The Eyes (Frozen CLIP)
        self.eyes = CLIPVisionEncoder(model_id=vision_id)

        # [3-1, 3-2단계: 뇌] The Brain (Reasoning LLM)
        self.brain = MistralLLM(
            model_id=llm_id,
            vision_hidden_size=1024,  # CLIP-Large Output
            llm_hidden_size=4096,     # Mistral Hidden
        )

        # [3-3단계: 뇌] The Translator (Spatial Reconstruction: Pixel Decoder)
        self.pixel_decoder = PixelDecoder(
            input_dim=4096,
            hidden_dim=512,
            output_dim=4096,
        )

        # [4단계: 컨트롤러] The Controller (FiLM Generator)
        self.controller = FiLMGenerator(
            input_dim=4096,
            vetnet_channels=vet_channels,  # 예: [64, 128, 256, 512]
        )

        # [5단계: 손] The Hands (Restoration Backbone: VETNet)
        self.hands = VETNet(
            in_channels=3,
            out_channels=3,
            dim=vet_channels[0],  # Base dim = 64
            # 나머지 Restormer/VETNet 파라미터는 기본값 사용 (필요 시 config에서 주입)
        )

        print("✅ DPR-Net V2 Assembly Complete!\n")

    def forward(self, batch):
        """
        Args:
            batch (dict): DataLoader에서 올라온 배치 데이터
                - pixel_values:   [B, 3, 224, 224]          (CLIP용)
                - input_ids:      [B, Text_Len]              (Mistral용)
                - attention_mask: [B, 257+Text_Len]          (Mistral용, Vision+Text)
                - high_res_images:[B, 3, H, W] or List[Tensor] (VETNet용)

        Returns:
            restored_image: [B, 3, H, W]
        """
        # 1. 데이터 언패킹
        clip_imgs = batch["pixel_values"]      # [B, 3, 224, 224]
        text_ids  = batch["input_ids"]         # [B, T]
        attn_mask = batch["attention_mask"]    # [B, 257+T] (설계 상)

        high_res = batch["high_res_images"]    # [B, 3, H, W] 또는 List[Tensor]

        # ----------------------------------------------------------------------
        # high_res 타입 처리 (Tensor or List[Tensor])
        # ----------------------------------------------------------------------
        if isinstance(high_res, list):
            # 리스트인 경우: 크기가 모두 같으면 stack, 아니면 batch_size=1만 허용
            if len(high_res) > 0 and isinstance(high_res[0], torch.Tensor):
                try:
                    high_res = torch.stack(high_res).to(clip_imgs.device)
                except RuntimeError:
                    if len(high_res) == 1:
                        high_res = high_res[0].unsqueeze(0).to(clip_imgs.device)
                    else:
                        raise ValueError(
                            "[DPRNetV2] high_res_images have different sizes. "
                            "Use batch_size=1 or pre-crop to a fixed size."
                        )
            else:
                raise ValueError("[DPRNetV2] high_res_images list is empty or not Tensor list.")
        elif isinstance(high_res, torch.Tensor):
            # Tensor인 경우: CLIP 이미지와 같은 디바이스로 맞춰줌
            high_res = high_res.to(clip_imgs.device)
        else:
            raise TypeError("[DPRNetV2] high_res_images must be Tensor or List[Tensor].")

        # ----------------------------------------------------------------------
        # Step 2: Eyes (Vision Encoding)
        # Output: [B, 257, 1024]
        # ----------------------------------------------------------------------
        vision_embeds = self.eyes(clip_imgs)  # (B, 257, 1024)

        # ----------------------------------------------------------------------
        # Step 3: Brain (Reasoning with Mistral)
        # Input:  vision_embeds [B, 257, 1024]
        #         text_ids      [B, T]
        #         attention_mask[B, 257+T]
        # Output: llm_output    [B, 257+T, 4096]
        # ----------------------------------------------------------------------
        B_v, L_v, _ = vision_embeds.shape
        B_t, L_t   = text_ids.shape
        B_m, L_m   = attn_mask.shape

        # 🔍 Sanity check: 배치 크기와 mask 길이 일치 여부
        assert B_v == B_t == B_m, \
            f"[DPRNetV2] Batch size mismatch: vision={B_v}, text={B_t}, mask={B_m}"
        expected_len = L_v + L_t
        assert L_m == expected_len, \
            f"[DPRNetV2] attention_mask length mismatch: expected {expected_len}, got {L_m}"

        llm_output = self.brain(vision_embeds, text_ids, attn_mask)  # (B, 257+T, 4096)

        # ----------------------------------------------------------------------
        # Step 4: Translator (Decoding Thought to Spatial Map)
        # PixelDecoder 내부에서 Vision Part (처음 257개)만 사용하여
        # [B, 257, 4096] -> [B, 257, 4096] (Refined)
        # ----------------------------------------------------------------------
        spatial_features = self.pixel_decoder(llm_output)  # (B, 257, 4096) 기대

        # ----------------------------------------------------------------------
        # Step 5: Controller (Generating FiLM Control Signals)
        # Input:  spatial_features [B, 257, 4096]
        # Output: film_signals: List[(gamma, beta), ...] for each VET stage
        # ----------------------------------------------------------------------
        film_signals = self.controller(spatial_features)

        # ----------------------------------------------------------------------
        # Step 6: Hands (Restoration with Volterra & FiLM)
        # Input:  high_res      [B, 3, H, W]
        #         film_signals  List[(gamma, beta), ...]
        # Output: restored_image [B, 3, H, W]
        # ----------------------------------------------------------------------
        restored_image = self.hands(high_res, film_signals)

        return restored_image

    # ==========================================================================
    # 🗣️ Extra: Chat Mode (For Debugging & Explanation)
    # 이미지를 보고 LLM이 뭐라고 생각했는지 텍스트로 물어보는 함수
    # ==========================================================================
    def chat_about_image(self, batch, max_new_tokens: int = 50):
        clip_imgs = batch["pixel_values"]
        text_ids  = batch["input_ids"]

        # CLIP으로 이미지 특징 추출
        vision_embeds = self.eyes(clip_imgs)  # (B, 257, 1024)

        # Mistral에게 텍스트 생성 요청 (설명 모드)
        captions = self.brain.generate_caption(
            vision_embeds,
            text_ids,
            max_new_tokens=max_new_tokens,
        )
        return captions
