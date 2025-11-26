import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
DATA_ROOT = r"G:\DPR-Net\data"
OUTPUT_JSON = os.path.join(DATA_ROOT, "metadata_captions.json")

# 처리할 데이터셋 폴더 목록
TARGET_FOLDERS = [
    "rain100H",    # Rain Removal
    "lol_dataset", # Low-Light Enhancement
    "CSD",         # Cloud/Snow Removal
    "SOTS"         # Dehazing
]

# ✅ 명세서에 정의된 "복원 최적화" 프롬프트
PROMPT_TEXT = "Question: Describe the weather conditions, lighting, and visual defects in this image detailedly. Answer:"

# 모델 설정 (메모리 부족 시 'Salesforce/blip2-opt-2.7b' 사용 권장)
MODEL_ID = "Salesforce/blip2-opt-2.7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================

def setup_blip2():
    """BLIP-2 모델과 프로세서를 로드합니다."""
    print(f"🚀 Loading BLIP-2 Model ({MODEL_ID}) on {DEVICE}...")
    print("   (This might take a few minutes for the first download)")
    
    processor = Blip2Processor.from_pretrained(MODEL_ID)
    
    # ⚡ FP16(반정밀도) 로드: 메모리 절약 및 속도 향상
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" # 가능한 경우 자동으로 GPU 분산 할당
    )
    
    return processor, model

def generate_caption(processor, model, image_path):
    """이미지 하나에 대해 캡션을 생성합니다."""
    try:
        # 이미지 로드 및 RGB 변환
        image = Image.open(image_path).convert('RGB')
        
        # 모델 입력 생성
        inputs = processor(images=image, text=PROMPT_TEXT, return_tensors="pt").to(DEVICE, torch.float16)
        
        # 캡션 생성 (Max tokens: 60 정도로 제한하여 핵심만 추출)
        generated_ids = model.generate(**inputs, max_new_tokens=60)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return caption
    except Exception as e:
        print(f"\n❌ Error processing {image_path}: {e}")
        return None

def get_all_image_paths(root_dir, target_folders):
    """지정된 폴더 내의 '손상된 이미지(Input)'만 똑똑하게 수집합니다."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_paths = []
    
    # 🚫 제외할 키워드 목록 (정답 이미지/GT 폴더명)
    # norain: 비 없는 정답
    # high: 밝은 정답 (LoL)
    # Gt: Ground Truth (CSD)
    # clear: 안개 없는 정답 (SOTS)
    # Mask: 마스크 파일
    IGNORE_KEYWORDS = ['norain', 'high', 'Gt', 'clear', 'Mask'] 
    
    print(f"🔍 Scanning directories in {root_dir}...")
    
    for folder in target_folders:
        full_path = os.path.join(root_dir, folder)
        if not os.path.exists(full_path):
            print(f"   ⚠️ Warning: Folder not found, skipping: {full_path}")
            continue
            
        for root, dirs, files in os.walk(full_path):
            # 현재 폴더 경로(root)에 제외 키워드가 하나라도 있으면 통째로 건너뜀
            # 예: "G:\...\rain100H\train\norain" -> 'norain'이 포함되어 있으므로 Skip
            if any(keyword in root for keyword in IGNORE_KEYWORDS):
                continue

            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
                    
    print(f"   ✅ Found {len(image_paths)} valid input images (filtered GT/Clean images).")
    return image_paths

def main():
    # 1. 모델 준비
    processor, model = setup_blip2()
    
    # 2. 기존 데이터 로드 (중단 후 이어하기 기능)
    metadata = {}
    if os.path.exists(OUTPUT_JSON):
        print(f"📂 Found existing metadata file: {OUTPUT_JSON}")
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"   resume from {len(metadata)} processed images.")
        except json.JSONDecodeError:
            print("   ⚠️ JSON file is corrupted. Starting from scratch.")

    # 3. 이미지 리스트 수집 (필터링 적용됨)
    image_files = get_all_image_paths(DATA_ROOT, TARGET_FOLDERS)
    
    # 4. 처리 루프
    save_interval = 50 # 50장마다 저장
    newly_processed_count = 0
    
    print("STARTING CAPTION GENERATION...")
    print("===================================================")
    
    for i, img_path in enumerate(tqdm(image_files)):
        # 이미 처리된 이미지는 스킵 (경로를 Key로 사용)
        # Windows 경로 호환성을 위해 정규화
        norm_path = os.path.normpath(img_path)
        
        # 기존 메타데이터에 키가 있는지 확인
        if norm_path in metadata or img_path in metadata:
            continue
            
        # 캡션 생성
        caption = generate_caption(processor, model, img_path)
        
        if caption:
            metadata[img_path] = caption # 원본 경로를 Key로 저장
            newly_processed_count += 1
            
        # 주기적 저장
        if newly_processed_count > 0 and newly_processed_count % save_interval == 0:
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
                
    # 5. 최종 저장
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    print("===================================================")
    print(f"🎉 DONE! Metadata saved to: {OUTPUT_JSON}")
    print(f"   Total processed: {len(metadata)}")

if __name__ == "__main__":
    main()