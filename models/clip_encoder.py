# G:\DPR-Net\models\clip_encoder.py
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

# ==============================================================================
# 👁️ The Eyes: CLIP Vision Encoder
# ==============================================================================

class CLIPVisionEncoder(nn.Module):
    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        super().__init__()
        print(f"👁️ Loading CLIP Vision Model: {model_id}...")

        self.vision_model = CLIPVisionModel.from_pretrained(model_id)

        # 🔒 Freeze
        self.vision_model.eval()
        for param in self.vision_model.parameters():
            param.requires_grad = False
            param.data = param.data.float()     # 🔥 AMP 시 안전

        # CLIP 공식 정규화
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

        print("   ✅ CLIP Model Frozen (FP32) & Normalization Ready.")

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [B, 3, 224, 224] (0~1 range recommended)
        Returns:
            image_embeds: [B, 257, 1024]
        """
        # 1) Normalize using CLIP official params (🔥 성능 향상 크다)
        pixel_values = (pixel_values - self.pixel_mean) / self.pixel_std

        # 2) Prevent AMP → keep visual features high precision
        with torch.cuda.amp.autocast(enabled=False):
            outputs = self.vision_model(pixel_values.float())

        # 3) Some CLIP versions do not expose last_hidden_state reliably
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]

        # 🔍 Sanity check (CLS + 256 patches)
        B, L, C = hidden.shape
        assert L == 257 and C == 1024, \
            f"[CLIPVisionEncoder] Expected (B,257,1024) got {hidden.shape}"

        return hidden  # (B, 257, 1024)
