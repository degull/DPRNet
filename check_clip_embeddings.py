import torch
import torch.nn.functional as F
from PIL import Image
import os

# --- 1. 모듈 임포트 ---
# (model 폴더에 __init__.py가 있다고 가정)
try:
    from model.dpr_modules import DprClipEncoder
except ImportError as e:
    print(f"Error: {e}")
    print("G:\\DPR-Net\\model\\__init__.py 파일이 있는지 확인하세요.")
    exit()

# --- 2. 헬퍼 함수 ---
def load_and_prep_image(path, clip_preprocess):
    """
    이미지 1장을 로드하고 CLIP이 요구하는
    (224x224 리사이즈, 정규화) 전처리를 수행합니다.
    """
    try:
        image = Image.open(path).convert("RGB")
        # DprClipEncoder에 포함된 preprocess 파이프라인 사용
        return clip_preprocess(image).unsqueeze(0) # (1, 3, 224, 224)
    except FileNotFoundError:
        print(f"\n[Error] 이미지를 찾을 수 없습니다: {path}")
        return None
    except Exception as e:
        print(f"\n[Error] 이미지 로드 중 오류: {e}")
        return None

def cosine_similarity(v1, v2):
    """두 텐서 벡터 간의 코사인 유사도 계산"""
    v1 = v1.flatten()
    v2 = v2.flatten()
    return F.cosine_similarity(v1, v2, dim=0).item()

# --- 3. 메인 테스트 ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- CLIP Embedding Meaningfulness Test (on {device}) ---")

    # --- 4. 모델 로드 ---
    # (dpr_modules.py의 테스트 코드는 실행 안 됨)
    print("Loading DprClipEncoder (Eyes v2)...")
    clip_encoder = DprClipEncoder().to(device)
    clip_encoder.eval()
    # 모델에 포함된 전처리기(resize, normalize 등)를 가져옴
    preprocess = clip_encoder.preprocess
    print("Encoder loaded.")

    # --- 5. 이미지 경로 설정 (사용자 데이터셋 기준) ---
    # (테스트할 이미지 1쌍의 경로를 지정하세요)
    clean_img_path = r"G:\DPR-Net\data\CSD\Test\Snow\10.tif"
    distorted_img_path = r"G:\DPR-Net\data\CSD\Test\Gt\10.tif"

    # --- 6. 이미지 로드 및 전처리 ---
    clean_tensor = load_and_prep_image(clean_img_path, preprocess)
    distorted_tensor = load_and_prep_image(distorted_img_path, preprocess)
    
    if clean_tensor is None or distorted_tensor is None:
        print("테스트를 중단합니다. 이미지 경로를 확인하세요.")
        exit()
        
    clean_tensor = clean_tensor.to(device)
    distorted_tensor = distorted_tensor.to(device)
    print(f"Loaded Clean Image: {clean_img_path}")
    print(f"Loaded Distorted Image: {distorted_img_path}")

    # --- 7. 임베딩 추출 ---
    with torch.no_grad():
        # (B, N+1, D), (B, D)
        tokens_clean, global_clean = clip_encoder(clean_tensor)
        tokens_distorted, global_distorted = clip_encoder(distorted_tensor)

    print("\n--- 1. Global Embedding Comparison ---")
    # (깨끗한 이미지 자신과 비교 = 1.0이 나와야 함)
    sim_self = cosine_similarity(global_clean, global_clean)
    print(f"Clean vs Clean (Self-Check): Similarity = {sim_self:.4f}")
    
    # (깨끗한 이미지 vs 왜곡된 이미지)
    sim_clean_vs_distorted = cosine_similarity(global_clean, global_distorted)
    print(f"Clean vs Distorted (Global): Similarity = {sim_clean_vs_distorted:.4f}")
    
    if sim_clean_vs_distorted < 0.99:
        print("  ✅ PASS: CLIP이 'Global' 수준에서 두 이미지의 차이를 감지했습니다.")
    else:
        print("  ⚠️ FAIL: CLIP이 두 이미지를 동일하게 보고 있습니다.")


    print("\n--- 2. Patch Embedding Comparison (Sample) ---")
    # (예: 50번째 패치 조각 비교)
    patch_clean_50 = tokens_clean[:, 50, :]
    patch_distorted_50 = tokens_distorted[:, 50, :]
    sim_patch_50 = cosine_similarity(patch_clean_50, patch_distorted_50)
    print(f"Patch #50 (Clean vs Distorted): Similarity = {sim_patch_50:.4f}")

    if sim_patch_50 < 0.99:
        print("  ✅ PASS: CLIP이 'Patch' 수준에서 두 이미지의 차이를 감지했습니다.")
    else:
        print("  ⚠️ FAIL: CLIP이 이 특정 패치를 동일하게 보고 있습니다.")