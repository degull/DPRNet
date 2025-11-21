import torch
import yaml
import os
from PIL import Image
import torchvision.transforms as T
from torch.cuda.amp import autocast

# --- 1. 모델 및 모듈 임포트 ---
try:
    from model.dpr_net import DPR_Net
except ImportError as e:
    print(f"Error: {e}")
    print("G:\\DPR-Net\\model\\__init__.py 파일이 있는지 확인하세요.")
    exit()

# --- 2. 헬퍼 함수 ---
def load_and_prep_image(path, device):
    """
    테스트용 이미지 1장을 로드하고 0~1 범위 텐서로 변환합니다.
    """
    try:
        image = Image.open(path).convert("RGB")
        transform = T.ToTensor()
        tensor = transform(image).unsqueeze(0) # (1, 3, H, W)
        return tensor.to(device)
    except FileNotFoundError:
        print(f"\n[Error] 이미지를 찾을 수 없습니다: {path}")
        return None
    except Exception as e:
        print(f"\n[Error] 이미지 로드 중 오류: {e}")
        return None

def decode_llm_explanation(generated_ids, tokenizer):
    """
    [수정] LLM이 생성한 'generated_ids'를 텍스트로 디코딩합니다.
    """
    # [0]번 배치의 텍스트만 디코딩
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text

# --- 3. 메인 평가 스크립트 ---
def main(config_path, checkpoint_path, test_image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- DPR-Net V2 Inference Test (on {device}) ---")

    # --- 1. 설정 로드 (YAML) ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return
        
    cfg_model_vet = config['model']['vetnet']
    cfg_model_clip = config['model']['clip']
    cfg_model_llm = config['model']['llm']

    # --- 2. 모델 초기화 (DPR_Net) ---
    print("Initializing REAL DPR-Net (V2) [LoRA Enabled] Model...")
    model = DPR_Net(
        vetnet_dim=cfg_model_vet['dim'],
        vetnet_num_blocks=cfg_model_vet['num_blocks'],
        vetnet_refinement_blocks=cfg_model_vet['refinement_blocks'],
        vetnet_heads=cfg_model_vet['heads'],
        vetnet_ffn_exp=cfg_model_vet['ffn_exp'],
        vetnet_bias=cfg_model_vet['bias'],
        vetnet_ln_type=cfg_model_vet['ln_type'],
        vetnet_volterra_rank=cfg_model_vet['volterra_rank'],
        clip_model_name=cfg_model_clip['model_name'],
        clip_pretrained=cfg_model_clip['pretrained'],
        clip_embed_dim=cfg_model_clip['embed_dim'],
        llm_model_name=cfg_model_llm['model_name'],
        llm_embed_dim=cfg_model_llm['embed_dim']
    ).to(device)
    
    model.eval() # (중요) 평가 모드로 설정
    
    # --- 3. 체크포인트 로드 ---
    if not os.path.isfile(checkpoint_path):
        print(f"[Error] 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # (LoRA 모델은 strict=False로 로드하는 것이 안전)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Checkpoint loaded successfully.")

    # --- 4. 테스트 이미지 로드 ---
    image_tensor = load_and_prep_image(test_image_path, device)
    if image_tensor is None:
        return
    print(f"Loaded test image: {test_image_path} (Shape: {image_tensor.shape})")

    # --- 5. 모델 추론 (Forward Pass) ---
    print("\nRunning model inference...")
    with torch.no_grad():
        # [수정] 8-bit 추론을 위해 autocast 래핑
        with autocast(): 
            outputs = model(image_tensor, mode='eval') # 'eval' 모드 명시
    print("Inference complete.")

    # --- 6. (A) Control Path 확인 ---
    print("\n--- (A) Control Path Verification ---")
    film_signals = outputs.get('film_signals')
    if film_signals and isinstance(film_signals, dict) and len(film_signals) == 8:
        print(f"  ✅ PASS: 8개의 FiLM 시그널 딕셔너리가 생성되었습니다.")
        print(f"     (예: 'latent' 신호 shape: {film_signals['latent'][0].shape})")
    else:
        print(f"  ⚠️ FAIL: Control Path 출력이 비정상적입니다. (dpr_net.py 수정 확인)")
        
    # --- 7. (B) Text Path 확인 ---
    print("\n--- (B) Text Path Verification ---")
    generated_ids = outputs.get('generated_ids')
    if generated_ids is not None:
        explanation_text = decode_llm_explanation(generated_ids, model.tokenizer)
        print(f"  ✅ PASS: LLM이 생성한 '설명 텍스트' (XAI):")
        print("  " + "="*40)
        print(f"  {explanation_text}")
        print("  " + "="*40)
    else:
        print(f"  ⚠️ FAIL: Text Path (generated_ids) 출력이 비정상적입니다.")
        
    # --- 8. (선택) 복원 이미지 저장 ---
    restored_img_tensor = outputs['img_restored'][0] 
    restored_img_pil = T.ToPILImage()(restored_img_tensor.cpu())
    
    output_path = "restored_output.png"
    restored_img_pil.save(output_path)
    print(f"\n✅ 복원된 이미지가 {output_path} 에 저장되었습니다.")


if __name__ == "__main__":
    # --- (설정) ---
    CONFIG_PATH = "configs/dpr_config.yaml"
    
    # [수정] 4 에포크 체크포인트로 변경
    CHECKPOINT_PATH = r"G:\DPR-Net\checkpoints_lora_en_2\epoch_9_total_0_0903_img_0_0823_text_0_0081_cons_0_0001.pth"
    
    # (테스트할 왜곡 이미지 1개 경로 - 예: Rain100H 테스트셋)
    TEST_IMAGE_PATH = r"G:\DPR-Net\data\SOTS\indoor\hazy\1403_5.png"
    # --- (설정 끝) ---

    # (경로에 폴더명이 checkpoints_lora가 맞는지 확인하세요)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[Error] 체크포인트 경로를 확인하세요: {CHECKPOINT_PATH}")
    # (테스트 이미지가 있는지 확인)
    elif not os.path.exists(TEST_IMAGE_PATH):
         print(f"[Error] 테스트 이미지 경로를 확인하세요: {TEST_IMAGE_PATH}")
    else:
        main(CONFIG_PATH, CHECKPOINT_PATH, TEST_IMAGE_PATH)