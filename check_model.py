import os
import sys
import torch
import torch.nn as nn

# 프로젝트 루트 경로 추가 (모듈 import 문제 방지)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dpr_net_v2 import DPRNetV2

# ==============================================================================
# 🛠️ Mock Configuration (테스트용 가짜 설정)
# 실제 config.yaml을 읽지 않고, 테스트에 필요한 최소 설정만 정의합니다.
# ==============================================================================
class TestConfig:
    def __init__(self):
        self.model = type('obj', (object,), {
            # 실제 모델 로드가 부담스럽다면 가벼운 모델로 교체해서 테스트 가능
            # "mistralai/Mistral-7B-v0.1" (실제) <-> "gpt2" (테스트용 가벼운 모델 - 구조 달라서 에러날 수 있음)
            # 여기서는 실제 통합 테스트를 위해 원래 ID 사용
            'llm_model_id': "mistralai/Mistral-7B-v0.1", 
            'vision_model_id': "openai/clip-vit-large-patch14",
            'vetnet_channels': [64, 128, 256, 512]
        })

def test_dpr_net():
    print("\n🚀 [Start] DPR-Net V2 Integration Test")
    
    # 1. 장치 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   - Device: {device}")
    
    # 2. 모델 초기화
    print("   - Initializing Model... (This mimics loading huge weights)")
    config = TestConfig()
    
    try:
        model = DPRNetV2(config).to(device)
        model.eval() # 평가 모드 (Dropout 등 비활성화)
        print("   ✅ Model initialized successfully!")
    except Exception as e:
        print(f"   ❌ Model Initialization Failed: {e}")
        return

    # 3. 더미 데이터 생성 (Batch Size = 2)
    print("\n📦 [Data] Generating Dummy Batch...")
    batch_size = 2
    text_len = 128
    vision_token_len = 257 # CLS + 256 Patches
    
    # (A) CLIP Input: [B, 3, 224, 224]
    pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # (B) Mistral Input IDs: [B, 128] (Random Integer Tokens)
    input_ids = torch.randint(0, 32000, (batch_size, text_len)).to(device)
    
    # (C) Attention Mask: [B, 257 + 128]
    # Vision(1) + Text(1) 형태
    vision_mask = torch.ones((batch_size, vision_token_len)).to(device)
    text_mask = torch.ones((batch_size, text_len)).to(device)
    attention_mask = torch.cat([vision_mask, text_mask], dim=1).long()
    
    # (D) VETNet Input (High-Res): [B, 3, 256, 256]
    # 실제 학습 시엔 Crop된 사이즈가 들어옴
    h, w = 256, 256
    high_res_images = torch.randn(batch_size, 3, h, w).to(device)
    
    batch = {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'high_res_images': high_res_images
    }
    
    print(f"   - Input Shapes:")
    print(f"     Pixel Values: {pixel_values.shape}")
    print(f"     Input IDs:    {input_ids.shape}")
    print(f"     Attn Mask:    {attention_mask.shape}")
    print(f"     High-Res Img: {high_res_images.shape}")

    # 4. Forward Pass (복원 모드 테스트)
    print("\n🔄 [Forward] Running Restoration Mode...")
    try:
        with torch.no_grad(): # 메모리 절약
            output = model(batch)
            
        print("   ✅ Forward Pass Successful!")
        print(f"   - Output Shape: {output.shape}")
        
        # 검증: 출력 크기가 입력(High-Res)과 같은지
        if output.shape == high_res_images.shape:
             print("   ✨ Shape Check Passed: Output matches Input Resolution.")
        else:
             print(f"   ⚠️ Shape Mismatch! Expected {high_res_images.shape}, got {output.shape}")

    except Exception as e:
        print(f"   ❌ Forward Pass Failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Chat Mode Test (설명 모드 테스트)
    print("\n💬 [Chat] Running Caption Generation Mode...")
    try:
        captions = model.chat_about_image(batch, max_new_tokens=20)
        print("   ✅ Chat Generation Successful!")
        print(f"   - Generated Captions (Random Init): {captions}")
    except Exception as e:
        print(f"   ❌ Chat Mode Failed: {e}")

    print("\n🏁 [End] Test Complete.")

if __name__ == "__main__":
    test_dpr_net()