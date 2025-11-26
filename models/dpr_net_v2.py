# 🏗️ 전체 조립 (Main Model)
import torch
import torch.nn as nn

# 각 모듈 가져오기
from models.clip_encoder import CLIPVisionEncoder
from models.mistral_llm import MistralLLM
from models.pixel_decoder import PixelDecoder
from models.film_layer import FiLMGenerator
from models.vetnet import VETNet

# ==============================================================================
# 🏗️ DPR-Net V2: The Final Assembly
# "From Auto-Captioning to Spatial Reconstruction"
#
# 구조:
# 1. Eyes (CLIP) -> 이미지 특징 추출 (Frozen)
# 2. Brain (Mistral) -> 텍스트+이미지 멀티모달 추론 (LoRA)
# 3. Translator (PixelDecoder) -> LLM 사고를 공간 맵으로 변환
# 4. Controller (FiLM) -> 복원 제어 신호(gamma, beta) 생성
# 5. Hands (VETNet) -> 고해상도 이미지 복원 수행
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
            # dict로 들어올 경우 편의를 위해 내부 변수로 할당
            llm_id = config['model']['llm_model_id']
            vision_id = config['model']['vision_model_id']
            vet_channels = config['model']['vetnet_channels']
        else:
            # Hydra/OmegaConf 등으로 로드된 객체일 경우
            llm_id = config.model.llm_model_id
            vision_id = config.model.vision_model_id
            vet_channels = config.model.vetnet_channels

        print("\n" + "="*60)
        print("🏗️ Initializing DPR-Net V2 (PixelLM Powered)...")
        print("="*60)

        # [2단계: 눈] The Eyes
        self.eyes = CLIPVisionEncoder(model_id=vision_id)

        # [3-1, 3-2단계: 뇌] The Brain (Reasoning)
        self.brain = MistralLLM(
            model_id=llm_id, 
            vision_hidden_size=1024, # CLIP-Large Output
            llm_hidden_size=4096     # Mistral Hidden
        )

        # [3-3단계: 뇌] The Translator (Spatial Reconstruction)
        self.pixel_decoder = PixelDecoder(
            input_dim=4096, 
            hidden_dim=512, 
            output_dim=4096
        )

        # [4단계: 컨트롤러] The Controller
        self.controller = FiLMGenerator(
            input_dim=4096, 
            vetnet_channels=vet_channels # [64, 128, 256, 512]
        )

        # [5단계: 손] The Hands (Restoration)
        # 사용자 정의 Volterra Layer가 포함된 VETNet
        self.hands = VETNet(
            in_channels=3, 
            out_channels=3, 
            dim=vet_channels[0], # Base dim = 64
            # 나머지 Restormer 파라미터는 기본값 사용 (필요 시 config에서 주입 가능)
        )
        
        print("✅ DPR-Net V2 Assembly Complete!\n")

    def forward(self, batch):
        """
        Args:
            batch (dict): DataLoader에서 올라온 배치 데이터
                - pixel_values: [B, 3, 224, 224] (CLIP용)
                - input_ids: [B, Text_Len] (Mistral용)
                - attention_mask: [B, 257+Text_Len] (Mistral용)
                - high_res_images: [B, 3, H, W] (VETNet용)
        
        Returns:
            restored_image: [B, 3, H, W]
        """
        # 1. 데이터 언패킹
        clip_imgs = batch['pixel_values']
        text_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        
        # VETNet 입력 이미지는 List[Tensor]일 수도 있고 Stacked Tensor일 수도 있음
        # 학습 시엔 Stacked Tensor, 추론 시엔 List일 수 있으므로 처리 필요
        high_res = batch['high_res_images']
        if isinstance(high_res, list):
            # 리스트라면(추론 시 이미지 크기가 다르면), 배치 처리를 위해 
            # 여기서는 일단 stack이 되어 있다고 가정하거나, 1개씩 처리해야 함.
            # * 학습 로더(dataset.py)에서 crop을 통해 크기를 맞춰서 stack해서 주는 것을 원칙으로 함.
            if len(high_res) > 0 and isinstance(high_res[0], torch.Tensor):
                 try:
                    high_res = torch.stack(high_res).to(clip_imgs.device)
                 except:
                    # 크기가 달라서 스택 불가 시, 배치 사이즈 1인 경우만 허용
                    if len(high_res) == 1:
                        high_res = high_res[0].unsqueeze(0).to(clip_imgs.device)
                    else:
                        raise ValueError("Batch processing requires images of same size. Use batch_size=1 for variable sizes.")
        
        # ----------------------------------------------------------------------
        # Step 2: Eyes (Vision Encoding)
        # Output: [B, 257, 1024]
        # ----------------------------------------------------------------------
        vision_embeds = self.eyes(clip_imgs)

        # ----------------------------------------------------------------------
        # Step 3: Brain (Reasoning)
        # Output: [B, 257+N, 4096]
        # ----------------------------------------------------------------------
        llm_output = self.brain(vision_embeds, text_ids, attn_mask)

        # ----------------------------------------------------------------------
        # Step 4: Translator (Decoding Thought to Spatial Map)
        # Output: [B, 257, 4096] (Vision Part Only, Refined)
        # ----------------------------------------------------------------------
        spatial_features = self.pixel_decoder(llm_output)

        # ----------------------------------------------------------------------
        # Step 5: Controller (Generating Control Signals)
        # Output: List of [(gamma, beta), ...]
        # ----------------------------------------------------------------------
        film_signals = self.controller(spatial_features)

        # ----------------------------------------------------------------------
        # Step 6: Hands (Restoration with Volterra & FiLM)
        # Output: [B, 3, H, W]
        # ----------------------------------------------------------------------
        restored_image = self.hands(high_res, film_signals)

        return restored_image

    # ==========================================================================
    # 🗣️ Extra: Chat Mode (For Debugging & Explanation)
    # 이미지를 보고 LLM이 뭐라고 생각했는지 텍스트로 물어보는 함수
    # ==========================================================================
    def chat_about_image(self, batch, max_new_tokens=50):
        clip_imgs = batch['pixel_values']
        text_ids = batch['input_ids']
        
        # CLIP으로 이미지 특징 추출
        vision_embeds = self.eyes(clip_imgs)
        
        # Mistral에게 텍스트 생성 요청
        captions = self.brain.generate_caption(
            vision_embeds, 
            text_ids, 
            max_new_tokens=max_new_tokens
        )
        return captions