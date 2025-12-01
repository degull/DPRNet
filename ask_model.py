import os
import torch
import yaml
from PIL import Image
from transformers import CLIPImageProcessor

# 사용자 모듈
from models.dpr_net_v2 import DPRNetV2

# ==============================================================================
# ⚙️ 설정
# ==============================================================================
CONFIG_PATH = "configs/dpr_config.yaml"

# 1. 확인하고 싶은 이미지 경로 (아까 노이즈 심했던 그 이미지 추천)
TEST_IMAGE_PATH = r"G:\DPR-Net\data\rain100H\train\rain\rain-001.png" 

# 2. 학습된 체크포인트 경로 (가장 최근에 저장된 것)
# 예: "G:\DPR-Net\logs\checkpoint_epoch_01_loss_....pth"
CHECKPOINT_PATH = r"G:\DPR-Net\logs\checkpoint_epoch_01_loss_0.5716_psnr_6.28_ssim_0.0000.pth" # <-- 파일명 수정 필요

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print(f"⚙️ Loading Config...")
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 빌드 (4-bit)
    print("🏗️ Loading Model...")
    model = DPRNetV2(config).to(device)
    
    # 2. 체크포인트 로드 (학습된 뇌 불러오기)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📂 Loading Trained Weights: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        # strict=False: 학습할 때 VETNet 파라미터 등은 텍스트 생성과 무관하므로 일부 불일치 허용
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("   ✅ Weights Loaded Successfully!")
    else:
        print("   ⚠️ Checkpoint not found! Using random weights (Just testing logic).")

    model.eval()

    # 3. 이미지 준비
    print(f"\n📸 Processing Image: {TEST_IMAGE_PATH}")
    clip_processor = CLIPImageProcessor.from_pretrained(config['model']['vision_model_id'])
    
    try:
        raw_image = Image.open(TEST_IMAGE_PATH).convert('RGB')
        pixel_values = clip_processor(images=raw_image, return_tensors="pt").pixel_values.to(device)
    except Exception as e:
        print(f"   ❌ Image load failed: {e}")
        return

    # 4. 질문 던지기 (프롬프트)
    # BLIP이 썼던 것과 비슷한 질문을 던져서, Mistral이 학습한 내용을 유도합니다.
    prompt_text = "Question: Describe the weather conditions and visual defects in this image detailedly. Answer:"
    
    tokenizer = model.brain.tokenizer
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    print(f"💬 Asking Mistral: '{prompt_text}'")
    print("-" * 50)

    # 5. 답변 생성 (Generate)
    with torch.no_grad():
        # dpr_net_v2.py에 chat_about_image 함수 활용
        # 내부적으로 model.brain.generate_caption을 호출함
        
        # 직접 brain의 generate 호출
        # (이미지 임베딩 추출 -> 텍스트와 결합 -> 생성)
        vision_embeds = model.eyes(pixel_values) # CLIP
        
        generated_text_list = model.brain.generate_caption(
            image_embeds=vision_embeds,
            input_ids=input_ids,
            max_new_tokens=100 # 최대 100단어 생성
        )
        
    # 6. 결과 출력
    print(f"🤖 Mistral's Thought:\n")
    print(f"\"{generated_text_list[0]}\"")
    print("-" * 50)

if __name__ == "__main__":
    main()