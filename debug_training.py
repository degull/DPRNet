import os
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# 사용자 모듈
from models.dpr_net_v2 import DPRNetV2
from data.dataset import DPRDataset

# ==============================================================================
# ⚙️ 설정
# ==============================================================================
CONFIG_PATH = "configs/dpr_config.yaml"
# 확인하고 싶은 체크포인트 경로
CHECKPOINT_PATH = r"G:\DPR-Net\logs\checkpoint_epoch_04_loss_0.1878_psnr_13.77_ssim_0.3949.pth"
OUTPUT_DIR = "debug_images"

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"⚙️ Loading Config...")
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 로드
    # (참고: 현재 4-bit 코드가 적용되어 있다면 4-bit로 로드됩니다. 
    # 체크포인트가 16-bit라면 strict=False로 어댑터만 로드해서 확인합니다.)
    try:
        print("🏗️ Loading Model...")
        model = DPRNetV2(config).to(device)
        
        print(f"📂 Loading Checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        # 키 불일치 무시하고 로드 (디버깅용)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("   ✅ Checkpoint loaded (strict=False)")
        
        model.eval()
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return

    # 2. 데이터셋 로드
    dataset = DPRDataset(
        data_root=config['paths']['root_dir'],
        metadata_file=config['paths']['metadata_file'],
        tokenizer_path=config['model']['llm_model_id'],
        max_length=config['model']['max_length']
    )
    
    # ⚡ [수정된 부분] collate_fn을 반드시 넣어줘야 attention_mask가 생성됩니다!
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=DPRDataset.collate_fn  # <--- 이거 추가함!
    )

    print("📸 Saving Debug Images...")
    
    with torch.no_grad():
        # autocast 추가 (4-bit 모델 호환성 위해)
        from torch.cuda.amp import autocast
        
        for i, batch in enumerate(tqdm(loader, total=5)):
            if i >= 5: break
            
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device) # 이제 에러 안 남
            vet_input = batch['vet_input'].to(device)
            vet_target = batch['vet_target'].to(device)

            with autocast():
                model_input = {
                    'pixel_values': pixel_values,
                    'input_ids': input_ids,
                    'attention_mask': attn_mask,
                    'high_res_images': vet_input
                }
                output = model(model_input)

            # 결과 저장 [Input (Noisy) | Output (Restored) | Target (GT)]
            # 값 범위 0~1로 클램핑
            vet_input = torch.clamp(vet_input, 0, 1)
            output = torch.clamp(output, 0, 1)
            vet_target = torch.clamp(vet_target, 0, 1)
            
            combined = torch.cat([vet_input, output, vet_target], dim=3)
            
            save_path = os.path.join(OUTPUT_DIR, f"debug_sample_{i}.png")
            save_image(combined, save_path)
            
    print(f"✅ Debug images saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("👉 폴더를 열어 이미지를 확인하세요. [왼쪽:입력 | 가운데:결과 | 오른쪽:정답]")

if __name__ == "__main__":
    main()