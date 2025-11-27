import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import time
import datetime

# Metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Modules
from models.dpr_net_v2 import DPRNetV2
from data.dataset import DPRDataset

CONFIG_PATH = "configs/dpr_config.yaml"

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def log_print(message, log_file):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def save_checkpoint(model, optimizer, epoch, loss, psnr, ssim, save_dir, log_file):
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
    log_print(f"\n💾 Checkpoint saved: {save_path}", log_file)

def main():
    # 1. Setup
    config = load_config(CONFIG_PATH)
    
    log_dir = config['paths']['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "training_log.txt")
    
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n{'='*60}", log_file)
    log_print(f"🚀 TRAINING STARTED AT: {start_time_str}", log_file)
    log_print(f"{'='*60}", log_file)
    
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    
    # 2. Dataset
    log_print("💿 Initializing Datasets...", log_file)
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
    log_print(f"   - Total Images: {len(dataset)}", log_file)
    
    # 3. Model (4-bit)
    log_print("🏗️ Building DPR-Net V2 Model (4-bit)...", log_file)
    model = DPRNetV2(config).to(device)
    
    # ⚡ [핵심 수정] 학습 가능한 파라미터는 무조건 FP32로 캐스팅 (GradScaler 에러 방지)
    log_print("🔧 Casting trainable parameters to Float32 for AMP stability...", log_file)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # 4. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=float(config['train']['learning_rate']), weight_decay=1e-4)
    
    criterion = nn.L1Loss()
    scaler = GradScaler()
    accum_steps = config['train']['accumulate_grad_batches']
    
    # Metrics
    metric_psnr = PeakSignalNoiseRatio().to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # 6. Training Loop
    num_epochs = config['train']['num_epochs']
    log_interval = 50 
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        t0 = time.time()
        
        log_print(f"\n[Epoch {epoch}/{num_epochs}] Started...", log_file)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for step, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            vet_input = batch['vet_input'].to(device)
            vet_target = batch['vet_target'].to(device)
            
            with autocast():
                model_input = {
                    'pixel_values': pixel_values,
                    'input_ids': input_ids,
                    'attention_mask': attn_mask,
                    'high_res_images': vet_input
                }
                
                restored_images = model(model_input)
                restored_clamped = torch.clamp(restored_images, 0.0, 1.0)
                
                loss = criterion(restored_images, vet_target)
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()
            
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            with torch.no_grad():
                current_loss = loss.item()
                batch_psnr = metric_psnr(restored_clamped, vet_target).item()
                batch_ssim = metric_ssim(restored_clamped, vet_target).item()
            
            epoch_loss += current_loss
            epoch_psnr += batch_psnr
            epoch_ssim += batch_ssim
            
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}", 
                'psnr': f"{batch_psnr:.2f}dB",
                'ssim': f"{batch_ssim:.4f}"
            })
            
            if (step + 1) % log_interval == 0:
                msg = f"   [Step {step+1}/{len(train_loader)}] Loss: {current_loss:.4f} | PSNR: {batch_psnr:.2f}dB | SSIM: {batch_ssim:.4f}"
                print(msg) 
        
        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)
        avg_ssim = epoch_ssim / len(train_loader)
        elapsed_time = time.time() - t0
        
        result_msg = (
            f"\n📊 Epoch {epoch} Done.\n"
            f"   Time: {elapsed_time:.2f}s\n"
            f"   Loss: {avg_loss:.5f}\n"
            f"   PSNR: {avg_psnr:.2f} dB\n"
            f"   SSIM: {avg_ssim:.4f}"
        )
        log_print(result_msg, log_file)
        save_checkpoint(model, optimizer, epoch, avg_loss, avg_psnr, avg_ssim, config['paths']['log_dir'], log_file)
        metric_psnr.reset()
        metric_ssim.reset()
        log_print("-" * 60, log_file)
    
    log_print("🏁 Training Finished Successfully!", log_file)

if __name__ == "__main__":
    main()