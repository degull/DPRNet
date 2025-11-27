import os
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast # ⚡ 핵심 추가: 혼합 정밀도 지원

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dpr_net_v2 import DPRNetV2

# ==============================================================================
# 🛠️ Mock Configuration (테스트용 가짜 설정)
# ==============================================================================
class TestConfig:
    def __init__(self):
        self.model = type('obj', (object,), {
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
        model.eval()
        
        # ⚠️ 중요: Mistral이 FP16이므로, Projector 등 새로 만든 레이어도 FP16으로 맞춰주는 것이 안전함
        # (학습 시에는 autocast가 해주지만, 명시적 변환이 더 확실함)
        model.brain.projector.half() 
        
        print("   ✅ Model initialized successfully!")
    except Exception as e:
        print(f"   ❌ Model Initialization Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 더미 데이터 생성
    print("\n📦 [Data] Generating Dummy Batch...")
    batch_size = 2
    text_len = 128
    vision_token_len = 257 
    
    # 입력 데이터는 기본적으로 FP32로 생성되지만, autocast 안에서 자동으로 처리됨
    pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    input_ids = torch.randint(0, 32000, (batch_size, text_len)).to(device)
    
    vision_mask = torch.ones((batch_size, vision_token_len)).to(device)
    text_mask = torch.ones((batch_size, text_len)).to(device)
    attention_mask = torch.cat([vision_mask, text_mask], dim=1).long()
    
    # VETNet 입력
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
        # ⚡ 핵심 수정: autocast 사용 (FP32 입력을 FP16 모델에 맞게 자동 변환)
        with torch.no_grad(), autocast(): 
            output = model(batch)
            
        print("   ✅ Forward Pass Successful!")
        print(f"   - Output Shape: {output.shape}")
        
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
        # Chat 모드도 autocast 필요
        with torch.no_grad(), autocast():
            # generate 함수 호출 시 attention_mask를 명시적으로 전달해야 경고가 사라짐
            # 하지만 chat_about_image 함수 내부를 수정하기보다는 여기서 예외처리만 확인
            captions = model.chat_about_image(batch, max_new_tokens=20)
            
        print("   ✅ Chat Generation Successful!")
        print(f"   - Generated Captions (Random Init): {captions}")
    except Exception as e:
        print(f"   ❌ Chat Mode Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🏁 [End] Test Complete.")

if __name__ == "__main__":
    test_dpr_net()