import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 사용자가 만든 모델 임포트
from model.dpr_net import DPR_Net

def visualize_brain_map(image_path, model_path=None, save_path="brain_heatmap.png"):
    print(f"--- Visualizing Spatial Map for: {image_path} ---")
    
    # 1. 설정 및 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 모델 초기화
    model = DPR_Net().to(device)
    model.eval() 
    
    if model_path:
        print(f"Loading weights from {model_path}...")
        # model.load_state_dict(torch.load(model_path)) 
    else:
        print("Warning: Using random initialized weights (Map will look random).")

    # 2. 이미지 전처리
    raw_image = Image.open(image_path).convert('RGB')
    raw_image_resized = raw_image.resize((224, 224)) 
    
    # 텐서 변환
    input_tensor = torch.from_numpy(np.array(raw_image_resized)).float() / 255.0
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device) 
    
    # CLIP Normalize
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    input_tensor = (input_tensor - mean) / std

    # 3. 모델 실행
    # [수정] 경고 메시지 해결을 위해 문법 수정 (torch.amp.autocast)
    with torch.no_grad():
        with torch.amp.autocast('cuda'): 
            vision_tokens, _ = model.clip_encoder(input_tensor)
            
            if model.training:
                _, features, _ = model.llm_module(vision_tokens, text_labels=None) 
            else:
                _, features, _ = model.llm_module(vision_tokens)

    # 4. 공간 지도 추출 및 가공 [여기가 에러 수정 핵심!]
    patch_features = features[:, 1:, :] # (1, 256, 4096)
    
    # Grid 변환
    grid_size = int(patch_features.shape[1] ** 0.5) # 16
    
    # [중요] OpenCV 호환성을 위해 float32로 명시적 변환 + NaN 제거
    feature_map_3d = patch_features.view(grid_size, grid_size, -1).float().cpu().numpy()
    
    # 채널 평균 (Activation Map)
    heatmap = np.mean(feature_map_3d, axis=2) 
    
    # [안전장치] 혹시 모를 무한대(inf)나 결측치(nan)를 0으로 치환
    heatmap = np.nan_to_num(heatmap)
    
    # [중요] OpenCV는 float32만 받음! 확실하게 변환
    heatmap = heatmap.astype(np.float32)

    # 5. 히트맵 그리기
    # 정규화 (0 ~ 1) - 분모가 0이 되는 것을 방지하기 위해 1e-8 추가
    min_val = heatmap.min()
    max_val = heatmap.max()
    heatmap = (heatmap - min_val) / (max_val - min_val + 1e-8)
    
    # Upsampling (이제 에러 안 날 겁니다)
    heatmap_resized = cv2.resize(heatmap, raw_image.size, interpolation=cv2.INTER_CUBIC)
    
    # Color Map
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Overlay
    orig_np = np.array(raw_image)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    alpha = 0.5 
    overlay = cv2.addWeighted(orig_np, 1 - alpha, heatmap_colored, alpha, 0)

    # 6. 결과 저장
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(raw_image)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title("Brain's Spatial Map (16x16)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay Result")
    plt.axis('off')
    
    plt.savefig(save_path)
    print(f"✅ Saved visualization to {save_path}")
    # plt.show() # (옵션) 서버 환경이면 이 줄 주석 처리

# visualize_map.py 맨 아래 실행 부분 수정
if __name__ == "__main__":
    TEST_IMAGE = "g:/DPR-Net/data/CSD/Train/Snow/2.tif" # 테스트 이미지
    
    # 방금 만든 가장 똑똑한 모델 경로
    MODEL_PATH = r"G:\DPR-Net\checkpoints\epoch_5_total_0_0722_img_0_0620_text_0_0096_cons_0_0558.pth"
    
    visualize_brain_map(TEST_IMAGE, model_path=MODEL_PATH)