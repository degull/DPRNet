import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# ==============================================================================
# 🧩 Custom Modules Import
# 프로젝트 구조에 맞게 모듈들을 가져옵니다.
# ==============================================================================
from models.dpr_net_v2 import DPRNetV2
from data.dataset import DPRDataset

# ==============================================================================
# ⚙️ Configuration Path
# ==============================================================================
CONFIG_PATH = "configs/dpr_config.yaml"

def load_config(path):
    """YAML 설정 파일을 로드합니다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """학습 중간 결과를 저장합니다."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"\n💾 Checkpoint saved: {save_path}")

def main():
    # --------------------------------------------------------------------------
    # 1. 초기 설정 (Setup)
    # --------------------------------------------------------------------------
    print(f"⚙️ Loading Configuration from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    
    # GPU 사용 가능 여부 확인
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    print(f"   - Device: {device}")
    
    # --------------------------------------------------------------------------
    # 2. 데이터셋 & 로더 준비 (Data Pipeline)
    # --------------------------------------------------------------------------
    print("💿 Initializing Datasets...")
    dataset = DPRDataset(
        data_root=config['paths']['root_dir'],
        metadata_file=config['paths']['metadata_file'],
        tokenizer_path=config['model']['llm_model_id'],
        max_length=config['model']['max_length']
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        num_workers=config['train']['num_workers'],
        collate_fn=DPRDataset.collate_fn, # 커스텀 배처 사용 필수
        pin_memory=True
    )
    print(f"   - Total Images: {len(dataset)}")
    print(f"   - Batch Size: {config['train']['batch_size']}")
    
    # --------------------------------------------------------------------------
    # 3. 모델 초기화 (Model Initialization)
    # --------------------------------------------------------------------------
    print("🏗️ Building DPR-Net V2 Model...")
    model = DPRNetV2(config).to(device)
    
    # --------------------------------------------------------------------------
    # 4. Optimizer 설정 (Parameter Filtering)
    # ⚠️ 중요: CLIP과 Mistral 본체는 Frozen 상태이므로 Optimizer에 등록하면 안 됨.
    # requires_grad=True인 파라미터(LoRA, Decoder, FiLM, VETNet)만 필터링합니다.
    # --------------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=float(config['train']['learning_rate']), weight_decay=1e-4)
    
    print(f"   - Trainable Parameters: {len(trainable_params)} tensors")
    
    # --------------------------------------------------------------------------
    # 5. Loss & Precision 설정
    # --------------------------------------------------------------------------
    # 복원 작업(Restoration)에는 L1 Loss(MAE)가 가장 효과적이고 안정적임
    criterion = nn.L1Loss()
    
    # Mixed Precision (FP16) - 메모리 절약 및 속도 향상
    scaler = GradScaler()
    
    # Gradient Accumulation - 적은 배치로 큰 배치 학습 효과를 냄
    accum_steps = config['train']['accumulate_grad_batches']
    
    # --------------------------------------------------------------------------
    # 6. 학습 루프 (Training Loop)
    # --------------------------------------------------------------------------
    print("\n🚀 STARTING TRAINING 🚀")
    print("="*60)
    
    num_epochs = config['train']['num_epochs']
    
    for epoch in range(1, num_epochs + 1):
        model.train() # 학습 모드 전환
        epoch_loss = 0.0
        
        # tqdm으로 진행률 표시
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # (A) 데이터 GPU로 이동
            pixel_values = batch['pixel_values'].to(device)       # CLIP 입력
            input_ids = batch['input_ids'].to(device)             # Mistral 입력
            attn_mask = batch['attention_mask'].to(device)        # Mistral 마스크
            vet_input = batch['vet_input'].to(device)             # VETNet 입력 (Noisy Image)
            vet_target = batch['vet_target'].to(device)           # 정답 (Ground Truth)
            
            # (B) Forward Pass (AutoCast 적용)
            # autocast 컨텍스트 내에서는 자동으로 FP16/FP32 연산을 섞어씀
            with autocast():
                # 모델이 기대하는 입력 딕셔너리 구성
                model_input = {
                    'pixel_values': pixel_values,
                    'input_ids': input_ids,
                    'attention_mask': attn_mask,
                    'high_res_images': vet_input
                }
                
                # 예측 결과 (Restored Image)
                restored_images = model(model_input)
                
                # Loss 계산 (예측 vs 정답)
                loss = criterion(restored_images, vet_target)
                
                # Gradient Accumulation을 위해 Loss를 나눔
                loss = loss / accum_steps

            # (C) Backward Pass
            # scaler를 사용하여 Loss 스케일링 후 역전파 (Underflow 방지)
            scaler.scale(loss).backward()
            
            # (D) Optimization Step (Accumulation 조건 만족 시)
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # 그래디언트 초기화
            
            # (E) Logging
            # 보여줄 때는 다시 스케일을 복구해서 표시
            current_loss = loss.item() * accum_steps
            epoch_loss += current_loss
            progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
        
        # Epoch 종료 후 평균 Loss 출력
        avg_loss = epoch_loss / len(train_loader)
        print(f"📊 Epoch {epoch} Done. Average Loss: {avg_loss:.5f}")
        
        # 체크포인트 저장
        save_checkpoint(model, optimizer, epoch, avg_loss, config['paths']['log_dir'])
        print("-" * 60)
    
    print("🏁 Training Finished Successfully!")

if __name__ == "__main__":
    main()