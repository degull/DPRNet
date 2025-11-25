import os
import yaml
import torch
import cv2
import numpy as np
import time
from PIL import Image # 이미지 로드용 추가
from torch.utils.data import DataLoader, Dataset

# --- 사용자 모듈 임포트 ---
try:
    from model.dpr_net import DPR_Net
except ImportError as e:
    print(f"Error: 패키지 로드 실패 ({e}).")
    exit(1)

# ----------------------------------------------------------
# 1. 사용자 지정 커스텀 데이터셋 클래스 (파일 리스트용)
# ----------------------------------------------------------
class CustomFileDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        
        # 이미지 로드 & 전처리 (0~1 float Tensor로 변환)
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            # (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        except Exception as e:
            print(f"이미지 로드 실패: {img_path} ({e})")
            # 에러 시 검은 화면 반환
            img_tensor = torch.zeros((3, 256, 256))

        # (Input Image, Dummy Target, File Name)
        # Target이 없으므로 Input을 그대로 Target 자리에 넣어서 리턴합니다.
        return img_tensor, img_tensor, os.path.basename(img_path)

# ----------------------------------------------------------
# 2. 유틸리티 함수
# ----------------------------------------------------------
def tensor2img(tensor):
    tensor = tensor.cpu().detach().numpy()
    if tensor.ndim == 4: tensor = tensor[0]
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = np.clip(tensor, 0, 1)
    return (tensor * 255.0).astype(np.uint8)

# ----------------------------------------------------------
# 3. 메인 실행 함수
# ----------------------------------------------------------
def evaluate_custom_files(config_path, checkpoint_path, target_files):
    print(f"--- 🔍 Custom File Evaluation ({len(target_files)} images) ---")
    
    # 1. 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 2. 모델 초기화
    print("Initializing Model...")
    cfg_model_vet = config['model']['vetnet']
    cfg_model_clip = config['model']['clip']
    cfg_model_llm = config['model']['llm']
    
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

    # 3. 체크포인트 로드
    print(f"Loading Checkpoint: {os.path.basename(checkpoint_path)}")
    if not os.path.exists(checkpoint_path):
        print("Error: 체크포인트 파일이 없습니다!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # module. 제거 등 키값 보정
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 4. 데이터 로더 설정 (사용자 지정 리스트 사용)
    dataset = CustomFileDataset(target_files)
    # num_workers=0 은 윈도우 필수
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 

    save_dir = "./results/custom_eval"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n--- 🚀 Loop Start (Real-time Log) ---")
    
    with torch.inference_mode():
        for i, batch in enumerate(data_loader):
            # 데이터 언패킹 (Input, Dummy, Filename)
            input_img, _, filename_tuple = batch
            filename = filename_tuple[0] # 튜플에서 파일명 추출
            
            print(f"\n[Image {i+1}/{len(target_files)}] : {filename}")
            
            # 1. Data Load Time
            t0 = time.time()
            input_img = input_img.to(device)
            t1 = time.time()
            # print(f"  > Data Loaded: {t1 - t0:.4f} sec")

            # 2. Inference Time
            print("  > Inferencing...")
            with torch.amp.autocast('cuda'): 
                outputs = model(input_img, mode='eval')
                restored_img = outputs['img_restored']
            t2 = time.time()
            print(f"  > Inference Done: {t2 - t1:.4f} sec")

            # 3. Save Time
            restored_np = tensor2img(restored_img)
            input_np = tensor2img(input_img)
            
            # 결과 저장: [원본] [복원] 나란히 붙이기
            save_filename = f"result_{filename}"
            comparison = np.hstack([input_np, restored_np])
            
            cv2.imwrite(os.path.join(save_dir, save_filename), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            t3 = time.time()
            print(f"  > Saved to {save_dir}/{save_filename}")

    print("\n--- Custom Evaluation Finished ---")
    print(f"Check images at: {save_dir}")

if __name__ == "__main__":
    CONFIG_PATH = "configs/dpr_config.yaml"
    
    # Epoch 5 체크포인트 경로 (본인 경로 확인!)
    CHECKPOINT_PATH = r"G:\DPR-Net\checkpoints\epoch_5_total_0_0722_img_0_0620_text_0_0096_cons_0_0558.pth"
    
    # ★ 여기에 테스트하고 싶은 이미지 경로를 리스트로 넣으세요 ★
    MY_TARGET_FILES = [
        r"G:\DPR-Net\data\rain100H\test\rain\norain-1.png",
        r"G:\DPR-Net\data\rain100H\test\rain\norain-2.png",
        r"G:\DPR-Net\data\rain100H\test\rain\norain-3.png",
        r"G:\DPR-Net\data\rain100H\test\rain\norain-4.png",
        r"G:\DPR-Net\data\rain100H\test\rain\norain-5.png"
    ]
    
    evaluate_custom_files(CONFIG_PATH, CHECKPOINT_PATH, MY_TARGET_FILES)