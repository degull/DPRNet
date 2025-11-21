# train.py
# batch_size = 1
""" import os
import yaml 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import open_clip
from torch.cuda.amp import autocast, GradScaler 

# --- 1. 모델 및 데이터셋 임포트 ---
try:
    from model.dpr_net import DPR_Net
    from data.dataset import DPRDataset
except ImportError as e:
    print(f"Error: 'model' 또는 'data' 패키지를 찾을 수 없습니다: {e}")
    print("G:\\DPR-Net\\model\\__init__.py 또는 G:\\DPR-Net\\data\\__init__.py 파일이 있는지 확인하세요.")
    exit(1)

# -----------------------------------------------
#  ⑦ V2 Training Loss (L_text + L_img + L_consistency)
# -----------------------------------------------
class DPRV2Loss(nn.Module):
    def __init__(self, 
                 lambda_consistency=0.01,
                 clip_model_name="ViT-L-14",
                 clip_pretrained="laion2b_s32b_b82k",
                 llm_embed_dim=4096
                 ):
        super().__init__()
        print(f"Initializing DPRV2Loss (L_img + L_consistency (λ={lambda_consistency}))")
        
        self.lambda_consistency = lambda_consistency
        self.l1_loss = nn.L1Loss()
        
        if self.lambda_consistency > 0:
            self.mse_loss = nn.MSELoss()
            print("  Loading CLIP Text Encoder for L_consistency...")
            self.clip_text_model, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=clip_pretrained
            )
            self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
            for param in self.clip_text_model.parameters():
                param.requires_grad = False
            print("  CLIP Text Encoder frozen.")
            try:
                clip_text_embed_dim = self.clip_text_model.text_projection.shape[1]
            except AttributeError:
                clip_text_embed_dim = self.clip_text_model.text_projection.out_features
            print(f"  Inferred CLIP Text embed dim: {clip_text_embed_dim}")
            self.consistency_projector = nn.Linear(llm_embed_dim, clip_text_embed_dim) 
        else:
            self.consistency_projector = None

    def calculate_consistency_loss(self, cls_hidden_state, logits, llm_tokenizer, device):
        llm_global_hidden = cls_hidden_state.squeeze(1) 
        projected_hidden = self.consistency_projector(llm_global_hidden.float())
        
        generated_tokens = torch.argmax(logits.detach(), dim=-1)
        generated_text_list = llm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        clip_text_tokens = self.clip_tokenizer(generated_text_list).to(device)
        
        with torch.no_grad(): 
            clip_text_embed = self.clip_text_model.encode_text(clip_text_tokens)
            
        target_text_embed = clip_text_embed.float().detach()
        loss_consistency = self.mse_loss(projected_hidden, target_text_embed)
        return loss_consistency

    def forward(self, outputs, target_img, llm_tokenizer, device):
        
        l_img = self.l1_loss(outputs['img_restored'], target_img)
        l_text = outputs['loss_text']
        
        l_consistency_val = 0
        if self.lambda_consistency > 0 and self.consistency_projector is not None:
            l_consistency = self.calculate_consistency_loss(
                outputs['cls_hidden_state'], 
                outputs['logits'], 
                llm_tokenizer,
                device
            )
            l_consistency_val = l_consistency.item()
        else:
            l_consistency = 0.0

        lambda_text = 1.0 
        total_loss = l_img + (lambda_text * l_text) + (self.lambda_consistency * l_consistency)
                     
        loss_dict = {
            'total': total_loss.item(),
            'L_img': l_img.item(),
            'L_text': l_text.item(),
            'L_cons': l_consistency_val
        }
        
        return total_loss, loss_dict


# -----------------------------------------------
#  🚀 메인 학습 스크립트 (train.py - LoRA 버전)
# -----------------------------------------------
def main(config_path):
    # --- 1. 설정 로드 (YAML) ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return
        
    cfg_paths = config['paths']
    cfg_train = config['train']
    cfg_loss = config['loss']
    cfg_model_vet = config['model']['vetnet']
    cfg_model_clip = config['model']['clip']
    cfg_model_llm = config['model']['llm']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(cfg_paths['checkpoint_dir'], exist_ok=True)

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
    
    tokenizer = model.tokenizer

    # --- 3. 데이터 로더 (DPRDataset) ---
    print(f"Loading REAL dataset from: {cfg_paths['data_dir']}")
    train_dataset = DPRDataset(
        data_dir=cfg_paths['data_dir'],
        tokenizer=tokenizer, 
        mode='train',
        patch_size=cfg_train['patch_size'],
        hybrid_prob=0.5,
        max_text_len=40 
    )
    
    num_workers = 4 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg_train['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 
    )
    
    # --- 4. 손실 함수 초기화 ---
    criterion = DPRV2Loss(
        lambda_consistency=cfg_loss.get('lambda_consistency', 0.01), 
        clip_model_name=cfg_model_clip['model_name'],
        clip_pretrained=cfg_model_clip['pretrained'],
        llm_embed_dim=cfg_model_llm['embed_dim']
    ).to(device)

    # --- 5. 옵티마이저 설정 ---
    #for name, param in model.vet_backbone.named_parameters():
    #    param.requires_grad = False
    
    params_to_train = list(filter(lambda p: p.requires_grad, model.parameters()))
    if criterion.consistency_projector is not None:
        params_to_train.extend(list(criterion.consistency_projector.parameters()))
        
    print(f"\nTotal trainable parameters (LoRA + Projectors ONLY): {sum(p.numel() for p in params_to_train):,}")
    
    optimizer = AdamW(params_to_train, lr=cfg_train['learning_rate'], weight_decay=1e-4)
    
    # --- 6. AMP 및 체크포인트 로드 ---
    scaler = GradScaler()
    max_grad_norm = 1.0 
    start_epoch = 0
    resume_path = cfg_paths.get('resume_checkpoint') 

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except RuntimeError as e:
            print(f"[Warning] Failed to load full state_dict (possibly due to PEFT structure): {e}")
            pass 
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] 
        print(f"Loaded checkpoint from epoch {start_epoch}. Resuming from epoch {start_epoch + 1}.")
    else:
        print("Starting training from scratch (Epoch 1).")

    # --- 7. 학습 루프 (LoRA + L_text) ---
    print(f"\n--- Starting DPR-Net V2 REAL Training (LoRA + L_text) ---")
    for epoch in range(start_epoch, cfg_train['epochs']):
        model.train() 
        criterion.train() 
        
        epoch_loss_total = 0.0
        epoch_loss_img = 0.0
        epoch_loss_text = 0.0
        epoch_loss_cons = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg_train['epochs']}")
        
        for distorted_img, clean_img, text_labels, text_mask in pbar:
            distorted_img = distorted_img.to(device)
            clean_img = clean_img.to(device)
            text_labels = text_labels.to(device)
            text_mask = text_mask.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(
                    distorted_img, 
                    text_labels=text_labels, 
                    text_attention_mask=text_mask,
                    mode='train'
                )

            loss, loss_dict = criterion(
                outputs, 
                clean_img, 
                model.tokenizer, 
                device
            )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(params_to_train, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss_total += loss.item()
            epoch_loss_img += loss_dict['L_img']
            epoch_loss_text += loss_dict['L_text']
            epoch_loss_cons += loss_dict['L_cons']

            pbar.set_postfix(
                total=f"{loss_dict['total']:.4f}",
                L_img=f"{loss_dict['L_img']:.4f}",
                L_text=f"{loss_dict['L_text']:.4f}", 
                L_cons=f"{loss_dict['L_cons']:.4f}"
            )
            
            if not torch.isfinite(loss):
                print(f"\n[Error] Loss became {loss.item()} at step. Stopping training.")
                return 
        
        num_batches = len(train_loader)
        avg_loss_total = epoch_loss_total / num_batches
        avg_loss_img = epoch_loss_img / num_batches
        avg_loss_text = epoch_loss_text / num_batches
        avg_loss_cons = epoch_loss_cons / num_batches
            
        print(f"Epoch {epoch+1} Average Loss: {avg_loss_total:.4f}")
        
        # --- 8. 체크포인트 저장 ---
        
        # [수정] 파일명에 L_cons 포함
        str_total = f"{avg_loss_total:.4f}".replace('.', '_')
        str_img = f"{avg_loss_img:.4f}".replace('.', '_')
        str_text = f"{avg_loss_text:.4f}".replace('.', '_')
        str_cons = f"{avg_loss_cons:.4f}".replace('.', '_') # <-- L_cons 추가
        
        filename = f"epoch_{epoch+1}_total_{str_total}_img_{str_img}_text_{str_text}_cons_{str_cons}.pth"
        checkpoint_path = os.path.join(cfg_paths['checkpoint_dir'], filename)
        
        save_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(), 
            'config': config 
        }
        
        torch.save(save_state, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("--- Training Finished ---")


if __name__ == "__main__":
    config_file_path = "configs/dpr_config.yaml" 
    main(config_file_path) """


# Gradient Accumulation
# train.py
import os
import yaml 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import open_clip
from torch.cuda.amp import autocast, GradScaler 

# --- 1. 모델 및 데이터셋 임포트 ---
try:
    from model.dpr_net import DPR_Net
    from data.dataset import DPRDataset
except ImportError as e:
    print(f"Error: 'model' 또는 'data' 패키지를 찾을 수 없습니다: {e}")
    print("G:\\DPR-Net\\model\\__init__.py 또는 G:\\DPR-Net\\data\\__init__.py 파일이 있는지 확인하세요.")
    exit(1)

# -----------------------------------------------
#  ⑦ V2 Training Loss (L_text + L_img + L_consistency)
# -----------------------------------------------
class DPRV2Loss(nn.Module):
    def __init__(self, 
                 lambda_consistency=0.01,
                 clip_model_name="ViT-L-14",
                 clip_pretrained="laion2b_s32b_b82k",
                 llm_embed_dim=4096
                 ):
        super().__init__()
        print(f"Initializing DPRV2Loss (L_img + L_consistency (λ={lambda_consistency}))")
        
        self.lambda_consistency = lambda_consistency
        self.l1_loss = nn.L1Loss()
        
        if self.lambda_consistency > 0:
            self.mse_loss = nn.MSELoss()
            print("  Loading CLIP Text Encoder for L_consistency...")
            self.clip_text_model, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=clip_pretrained
            )
            self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
            for param in self.clip_text_model.parameters():
                param.requires_grad = False
            print("  CLIP Text Encoder frozen.")
            try:
                clip_text_embed_dim = self.clip_text_model.text_projection.shape[1]
            except AttributeError:
                clip_text_embed_dim = self.clip_text_model.text_projection.out_features
            print(f"  Inferred CLIP Text embed dim: {clip_text_embed_dim}")
            self.consistency_projector = nn.Linear(llm_embed_dim, clip_text_embed_dim) 
        else:
            self.consistency_projector = None

    def calculate_consistency_loss(self, cls_hidden_state, logits, llm_tokenizer, device):
        llm_global_hidden = cls_hidden_state.squeeze(1) 
        projected_hidden = self.consistency_projector(llm_global_hidden.float())
        
        generated_tokens = torch.argmax(logits.detach(), dim=-1)
        generated_text_list = llm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        clip_text_tokens = self.clip_tokenizer(generated_text_list).to(device)
        
        with torch.no_grad(): 
            clip_text_embed = self.clip_text_model.encode_text(clip_text_tokens)
            
        target_text_embed = clip_text_embed.float().detach()
        loss_consistency = self.mse_loss(projected_hidden, target_text_embed)
        return loss_consistency

    def forward(self, outputs, target_img, llm_tokenizer, device):
        
        l_img = self.l1_loss(outputs['img_restored'], target_img)
        l_text = outputs['loss_text']
        
        l_consistency_val = 0
        if self.lambda_consistency > 0 and self.consistency_projector is not None:
            l_consistency = self.calculate_consistency_loss(
                outputs['cls_hidden_state'], 
                outputs['logits'], 
                llm_tokenizer,
                device
            )
            l_consistency_val = l_consistency.item()
        else:
            l_consistency = 0.0

        lambda_text = 1.0 
        total_loss = l_img + (lambda_text * l_text) + (self.lambda_consistency * l_consistency)
                      
        loss_dict = {
            'total': total_loss.item(),
            'L_img': l_img.item(),
            'L_text': l_text.item(),
            'L_cons': l_consistency_val
        }
        
        return total_loss, loss_dict


# -----------------------------------------------
#  🚀 메인 학습 스크립트 (train.py - LoRA 버전 + Gradient Accumulation)
# -----------------------------------------------
def main(config_path):
    # --- 1. 설정 로드 (YAML) ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return
        
    cfg_paths = config['paths']
    cfg_train = config['train']
    cfg_loss = config['loss']
    cfg_model_vet = config['model']['vetnet']
    cfg_model_clip = config['model']['clip']
    cfg_model_llm = config['model']['llm']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(cfg_paths['checkpoint_dir'], exist_ok=True)

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
    
    tokenizer = model.tokenizer

    # --- 3. 데이터 로더 (DPRDataset) ---
    print(f"Loading REAL dataset from: {cfg_paths['data_dir']}")
    train_dataset = DPRDataset(
        data_dir=cfg_paths['data_dir'],
        tokenizer=tokenizer, 
        mode='train',
        patch_size=cfg_train['patch_size'],
        hybrid_prob=0.5,
        max_text_len=40 
    )
    
    num_workers = 4 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg_train['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 
    )
    
    # --- 4. 손실 함수 초기화 ---
    criterion = DPRV2Loss(
        lambda_consistency=cfg_loss.get('lambda_consistency', 0.01), 
        clip_model_name=cfg_model_clip['model_name'],
        clip_pretrained=cfg_model_clip['pretrained'],
        llm_embed_dim=cfg_model_llm['embed_dim']
    ).to(device)

    # --- 5. 옵티마이저 설정 ---
    # [중요] VETNet 학습을 위해 아래 코드는 주석 처리 유지 또는 삭제
    # for name, param in model.vet_backbone.named_parameters():
    #     param.requires_grad = False
    
    params_to_train = list(filter(lambda p: p.requires_grad, model.parameters()))
    if criterion.consistency_projector is not None:
        params_to_train.extend(list(criterion.consistency_projector.parameters()))
        
    print(f"\nTotal trainable parameters (LoRA + Projectors ONLY): {sum(p.numel() for p in params_to_train):,}")
    
    optimizer = AdamW(params_to_train, lr=cfg_train['learning_rate'], weight_decay=1e-4)
    
    # --- 6. AMP 및 체크포인트 로드 ---
    scaler = GradScaler()
    max_grad_norm = 1.0 
    start_epoch = 0
    resume_path = cfg_paths.get('resume_checkpoint') 

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except RuntimeError as e:
            print(f"[Warning] Failed to load full state_dict (possibly due to PEFT structure): {e}")
            pass 
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] 
        print(f"Loaded checkpoint from epoch {start_epoch}. Resuming from epoch {start_epoch + 1}.")
    else:
        print("Starting training from scratch (Epoch 1).")

    # --- 7. 학습 루프 (LoRA + L_text + Gradient Accumulation) ---
    
    # [추가] Accumulation Steps 설정 (기본값 4)
    accumulation_steps = cfg_train.get('accumulation_steps', 4)
    print(f"\n--- Starting DPR-Net V2 REAL Training (LoRA + L_text) ---")
    print(f"Batch Size: {cfg_train['batch_size']} | Accumulation Steps: {accumulation_steps}")
    print(f"Effective Batch Size: {cfg_train['batch_size'] * accumulation_steps}")
    
    

    for epoch in range(start_epoch, cfg_train['epochs']):
        model.train() 
        criterion.train() 
        
        epoch_loss_total = 0.0
        epoch_loss_img = 0.0
        epoch_loss_text = 0.0
        epoch_loss_cons = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg_train['epochs']}")
        
        # [수정] 루프 시작 전 그래디언트 초기화
        optimizer.zero_grad()
        
        for i, (distorted_img, clean_img, text_labels, text_mask) in enumerate(pbar):
            distorted_img = distorted_img.to(device)
            clean_img = clean_img.to(device)
            text_labels = text_labels.to(device)
            text_mask = text_mask.to(device)
            
            # [주의] 여기서 optimizer.zero_grad()를 호출하면 안 됩니다! (Accumulation 중이므로)
            
            with autocast():
                outputs = model(
                    distorted_img, 
                    text_labels=text_labels, 
                    text_attention_mask=text_mask,
                    mode='train'
                )

            loss, loss_dict = criterion(
                outputs, 
                clean_img, 
                model.tokenizer, 
                device
            )
            
            # [수정] Loss Normalize: Accumulation Steps만큼 나눠줍니다.
            loss = loss / accumulation_steps
            
            # Backward (매번 실행해서 그래디언트를 그릇에 모음)
            scaler.scale(loss).backward()
            
            # [수정] Step (정해진 횟수만큼 모였을 때만 실행)
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(params_to_train, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # 업데이트 후 그릇 비우기
            
            # 로그 기록 (나누어진 loss를 다시 곱해서 원래 크기로 표시)
            loss_val_for_log = loss.item() * accumulation_steps
            
            epoch_loss_total += loss_val_for_log
            epoch_loss_img += loss_dict['L_img']
            epoch_loss_text += loss_dict['L_text']
            epoch_loss_cons += loss_dict['L_cons']

            pbar.set_postfix(
                            total=f"{loss_val_for_log:.4f}",
                            L_img=f"{loss_dict['L_img']:.4f}",
                            L_text=f"{loss_dict['L_text']:.4f}",
                            L_cons=f"{loss_dict['L_cons']:.4f}",
                            step=f"{(i+1)%accumulation_steps}/{accumulation_steps}"
                        )
            
            if not torch.isfinite(loss):
                print(f"\n[Error] Loss became {loss.item()} at step. Stopping training.")
                return 
        
        num_batches = len(train_loader)
        avg_loss_total = epoch_loss_total / num_batches
        avg_loss_img = epoch_loss_img / num_batches
        avg_loss_text = epoch_loss_text / num_batches
        avg_loss_cons = epoch_loss_cons / num_batches
            
        print(f"Epoch {epoch+1} Average Loss: {avg_loss_total:.4f}")
        
        # --- 8. 체크포인트 저장 ---
        str_total = f"{avg_loss_total:.4f}".replace('.', '_')
        str_img = f"{avg_loss_img:.4f}".replace('.', '_')
        str_text = f"{avg_loss_text:.4f}".replace('.', '_')
        str_cons = f"{avg_loss_cons:.4f}".replace('.', '_')
        
        filename = f"epoch_{epoch+1}_total_{str_total}_img_{str_img}_text_{str_text}_cons_{str_cons}.pth"
        checkpoint_path = os.path.join(cfg_paths['checkpoint_dir'], filename)
        
        save_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(), 
            'config': config 
        }
        
        torch.save(save_state, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("--- Training Finished ---")


if __name__ == "__main__":
    config_file_path = "configs/dpr_config.yaml" 
    main(config_file_path)