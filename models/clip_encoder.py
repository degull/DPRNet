# 눈 (Frozen CLIP)
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

# ==============================================================================
# 👁️ The Eyes: CLIP Vision Encoder
# 역할: 이미지를 보고 257개의 의미론적 토큰(CLS + Patches)으로 변환합니다.
# 특징: 학습되지 않도록 Frozen 상태를 유지하여 강력한 특징 추출 능력을 보존합니다.
# ==============================================================================

class CLIPVisionEncoder(nn.Module):
    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        """
        Args:
            model_id (str): Hugging Face Model ID (기본값: ViT-Large/14)
        """
        super().__init__()
        print(f"👁️ Loading CLIP Vision Model: {model_id}...")
        
        # CLIP의 Vision Encoder 부분만 로드합니다.
        # (Text Encoder는 사용하지 않으므로 메모리 절약)
        self.vision_model = CLIPVisionModel.from_pretrained(model_id)
        
        # 🧊 Freeze Parameters (학습 방지)
        # CLIP은 이미지를 보는 법을 이미 잘 알고 있으므로, 
        # 이 가중치가 학습 중에 망가지지 않도록 고정합니다.
        self.vision_model.eval() # 평가 모드 고정
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        print("   ✅ CLIP Model Frozen successfully.")

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [Batch, 3, 224, 224] - 정규화된 이미지 텐서
            
        Returns:
            image_embeds: [Batch, 257, 1024] - (1 CLS Token + 256 Patch Tokens)
        """
        # CLIP Vision Model Forward Pass
        # output_hidden_states=True를 하지 않아도 기본적으로 last_hidden_state는 반환됨
        outputs = self.vision_model(pixel_values=pixel_values)
        
        # ⚠️ 중요: pooler_output이 아닌 last_hidden_state를 사용해야 함!
        # pooler_output: [Batch, 1024] -> 이미지를 1개의 벡터로 압축해버림 (공간 정보 소실)
        # last_hidden_state: [Batch, 257, 1024] -> 패치별 정보가 살아있음 (복원에 필수)
        return outputs.last_hidden_state