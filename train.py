import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import time

# ------------------------------------------------------------------------------
# 📊 Metrics Library Import
# ------------------------------------------------------------------------------
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Custom Modules
from models.dpr_net_v2 import DPRNetV2
from data.dataset import DPRDataset

# ==============================================================================
# ⚙️ Configuration Path
# ==============================================================================
CONFIG_PATH = "configs/dpr_config.yaml"

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch, loss, psnr, ssim, save_dir):
    """
    파일명에 Loss, PSNR, SSIM을 모두 기록합니다.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"checkpoint_epoch_{epoch:02d}_loss_{loss:.4f}_psnr_{psnr:.2f}_ssim_{ssim:.4f}.pth"
    save_path = os.path.join(save_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'psnr': psnr,
        'ssim': ssim
    }, save_path)
    print(f"\n💾 Checkpoint saved: {save_path}")

def main():
    # --------------------------------------------------------------------------
    # 1. 초기 설정
    # --------------------------------------------------------------------------
    print(f"⚙️ Loading Configuration from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    print(f"   - Device: {device}")
    
    # --------------------------------------------------------------------------
    # 2. 데이터셋 & 로더
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
        collate_fn=DPRDataset.collate_fn,
        pin_memory=True
    )
    print(f"   - Total Images: {len(dataset)}")
    print(f"   - Batch Size: {config['train']['batch_size']}")
    
    # --------------------------------------------------------------------------
    # 3. 모델 초기화
    # --------------------------------------------------------------------------
    print("🏗️ Building DPR-Net V2 Model...")
    model = DPRNetV2(config).to(device)
    
    # ⚠️ Windows 호환성을 위해 torch.compile 제거 (안정성 최우선)
    # print("🚀 Compiling model...") -> 삭제함
    
    # --------------------------------------------------------------------------
    # 4. Optimizer & Metrics
    # --------------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=float(config['train']['learning_rate']), weight_decay=1e-4)
    
    criterion = nn.L1Loss()
    scaler = GradScaler()
    accum_steps = config['train']['accumulate_grad_batches']
    
    # Metrics (GPU)
    metric_psnr = PeakSignalNoiseRatio().to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # --------------------------------------------------------------------------
    # 6. 학습 루프
    # --------------------------------------------------------------------------
    print("\n🚀 STARTING TRAINING (Stable Mode) 🚀")
    print("="*60)
    
    num_epochs = config['train']['num_epochs']
    log_interval = 100 
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        
        start_time = time.time()
        
        print(f"\n[Epoch {epoch}/{num_epochs}] Started...")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for step, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            vet_input = batch['vet_input'].to(device)
            vet_target = batch['vet_target'].to(device)
            
            # Forward Pass
            with autocast():
                model_input = {
                    'pixel_values': pixel_values,
                    'input_ids': input_ids,
                    'attention_mask': attn_mask,
                    'high_res_images': vet_input
                }
                
                restored_images = model(model_input)
                
                # Loss 계산
                loss = criterion(restored_images, vet_target)
                loss_scaled = loss / accum_steps

            # Backward Pass
            scaler.scale(loss_scaled).backward()
            
            # Optimization
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Metrics (no_grad)
            with torch.no_grad():
                restored_clamped = torch.clamp(restored_images, 0.0, 1.0)
                current_loss = loss.item()
                
                batch_psnr = metric_psnr(restored_clamped, vet_target).item()
                batch_ssim = metric_ssim(restored_clamped, vet_target).item()
            
            # 누적
            epoch_loss += current_loss
            epoch_psnr += batch_psnr
            epoch_ssim += batch_ssim
            
            # Progress Bar
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}", 
                'psnr': f"{batch_psnr:.2f}dB"
            })
            
            # 터미널 로그
            if (step + 1) % log_interval == 0:
                print(f"   [Step {step+1}/{len(train_loader)}] "
                      f"Loss: {current_loss:.4f} | PSNR: {batch_psnr:.2f} | SSIM: {batch_ssim:.4f}")
        
        # Epoch 결과 집계
        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)
        avg_ssim = epoch_ssim / len(train_loader)
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 Epoch {epoch} Done.")
        print(f"   Time: {elapsed_time:.2f}s")
        print(f"   Loss: {avg_loss:.5f}")
        print(f"   PSNR: {avg_psnr:.2f} dB")
        print(f"   SSIM: {avg_ssim:.4f}")
        
        # 체크포인트 저장
        save_checkpoint(model, optimizer, epoch, avg_loss, avg_psnr, avg_ssim, config['paths']['log_dir'])
        
        # Metric 초기화
        metric_psnr.reset()
        metric_ssim.reset()
        
        print("-" * 60)
    
    print("🏁 Training Finished Successfully!")

if __name__ == "__main__":
    main()