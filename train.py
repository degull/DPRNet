import os
import yaml  # PyYAML (pip install PyYAML)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import open_clip
# --- [수정 1] AMP(Automatic Mixed Precision) 모듈 임포트 ---
from torch.cuda.amp import autocast, GradScaler 

# --- 1. 모델 및 데이터셋 임포트 ---
try:
    # 실제 모델 및 모듈 임포트
    from model.dpr_net import DPR_Net
    from model.dpr_modules import DprLLM # (DPRV2Loss가 LLM 토크나이저 필요)
    from data.dataset import DPRDataset
except ImportError as e:
    print(f"Error: 'model' 또는 'data' 패키지를 찾을 수 없습니다: {e}")
    print("G:\\DPR-Net\\model\\__init__.py 또는 G:\\DPR-Net\\data\\__init__.py 파일이 있는지 확인하세요.")
    exit(1)

# -----------------------------------------------
#  ⑦ V2 Training Loss (L_total)
#  (이 클래스는 수정할 필요 없습니다. autocast가 자동으로 처리합니다.)
# -----------------------------------------------
class DPRV2Loss(nn.Module):
    """
    DPR-Net V2의 ⑦번 전체 손실 함수
    L_total = L_img + λ1 * L_film + λ2 * L_consistency
    """
    def __init__(self, 
                 lambda_film=0.1, 
                 lambda_consistency=0.01,
                 clip_model_name="ViT-L-14",
                 clip_pretrained="laion2b_s32b_b82k",
                 llm_embed_dim=4096
                 ):
        super().__init__()
        print(f"Initializing DPRV2Loss (λ_film={lambda_film}, λ_consistency={lambda_consistency})")
        
        self.lambda_film = lambda_film
        self.lambda_consistency = lambda_consistency

        # --- L_img (복원 손실) ---
        self.l1_loss = nn.L1Loss()
        
        # --- L_consistency (일관성 손실) ---
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

    def calculate_film_loss(self, hidden_state):
        num_tokens_to_align = 8 
        if hidden_state.shape[1] < num_tokens_to_align:
            return 0 
            
        control_tokens = hidden_state[:, :num_tokens_to_align, :]
        tokens_norm = F.normalize(control_tokens, p=2, dim=2)
        cosine_sim_matrix = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))
        eye = torch.eye(num_tokens_to_align, device=cosine_sim_matrix.device).expand_as(cosine_sim_matrix)
        off_diagonal_sim = cosine_sim_matrix * (1 - eye)
        loss_film = torch.mean(off_diagonal_sim.pow(2))
        return loss_film

    def calculate_consistency_loss(self, hidden_state, logits, llm_tokenizer, device):
        llm_global_hidden = hidden_state[:, 0, :] 
        projected_hidden = self.consistency_projector(llm_global_hidden)
        
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
        
        l_film_val = 0
        if self.lambda_film > 0:
            l_film = self.calculate_film_loss(outputs['hidden_state'])
            l_film_val = l_film.item()
        else:
            l_film = 0.0

        l_consistency_val = 0
        if self.lambda_consistency > 0:
            l_consistency = self.calculate_consistency_loss(
                outputs['hidden_state'], 
                outputs['logits'], 
                llm_tokenizer,
                device
            )
            l_consistency_val = l_consistency.item()
        else:
            l_consistency = 0.0

        total_loss = l_img + \
                     self.lambda_film * l_film + \
                     self.lambda_consistency * l_consistency
                     
        loss_dict = {
            'total': total_loss.item(),
            'img': l_img.item(),
            'film': l_film_val,
            'consistency': l_consistency_val
        }
        
        return total_loss, loss_dict


# -----------------------------------------------
#  🚀 메인 학습 스크립트 (train.py)
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
    
    # --- 2. 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(cfg_paths['checkpoint_dir'], exist_ok=True)

    # --- 3. 데이터 로더 (DPRDataset 사용) ---
    print(f"Loading REAL dataset from: {cfg_paths['data_dir']}")
    train_dataset = DPRDataset(
        data_dir=cfg_paths['data_dir'],
        mode='train',
        patch_size=cfg_train['patch_size'],
        hybrid_prob=0.5 
    )
    
    num_workers = 4 # (시스템에 맞게 조절, 0도 가능)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg_train['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        # --- [수정 2] 데이터 로더 최적화 ---
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 # (선택 사항, 데이터 준비 속도 향상)
    )
    
    # --- 4. 모델 초기화 (DPR_Net 사용) ---
    print("Initializing REAL DPR-Net (V2) Model...")
    # (Mistral 다운로드는 이미 완료되었을 것입니다)
    
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
    
    # --- 5. 손실 함수 초기화 ---
    criterion = DPRV2Loss(
        lambda_film=cfg_loss['lambda_film'],
        lambda_consistency=cfg_loss['lambda_consistency'],
        clip_model_name=cfg_model_clip['model_name'],
        clip_pretrained=cfg_model_clip['pretrained'],
        llm_embed_dim=cfg_model_llm['embed_dim']
    ).to(device)

    # --- 6. 옵티마이저 설정 ---
    params_to_train = list(filter(lambda p: p.requires_grad, model.parameters()))
    params_to_train.extend(list(criterion.consistency_projector.parameters()))
    
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in params_to_train):,}")
    
    optimizer = AdamW(params_to_train, lr=cfg_train['learning_rate'], weight_decay=1e-4)
    
    # --- [수정 3] GradScaler 초기화 ---
    scaler = GradScaler()
    
    # --- 7. 학습 루프 ---
    print(f"\n--- Starting DPR-Net V2 REAL Training (AMP Enabled) ---")
    for epoch in range(cfg_train['epochs']):
        model.train() 
        criterion.consistency_projector.train()
        
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg_train['epochs']}")
        
        for distorted_img, clean_img in pbar:
            distorted_img = distorted_img.to(device)
            clean_img = clean_img.to(device)
            
            # (옵티마이저 초기화)
            optimizer.zero_grad()
            
            # --- [수정 4] autocast로 Forward Pass 래핑 ---
            # (float16으로 연산)
            with autocast():
                # 1. Forward pass
                outputs = model(distorted_img)
                
                # 2. Calculate V2 Loss
                loss, loss_dict = criterion(
                    outputs, 
                    clean_img, 
                    model.tokenizer, # (DPR_Net이 소유한) LLM 토크나이저
                    device
                )
            
            # --- [수정 5] Scaler를 사용한 Backward Pass ---
            scaler.scale(loss).backward()
            
            # --- [수정 6] Scaler를 사용한 Optimizer Step ---
            scaler.step(optimizer)
            
            # --- [수정 7] Scaler 업데이트 ---
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(
                total=f"{loss_dict['total']:.4f}",
                img=f"{loss_dict['img']:.4f}",
                film=f"{loss_dict['film']:.4f}",
                cons=f"{loss_dict['consistency']:.4f}"
            )
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}")
        
        # --- 8. 체크포인트 저장 ---
        if (epoch + 1) % cfg_train['save_every'] == 0:
            checkpoint_path = os.path.join(cfg_paths['checkpoint_dir'], f"dpr_net_epoch_{epoch+1}.pth")
            
            save_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(), # --- [수정 8] Scaler 상태 저장
                'config': config # 설정 파일도 함께 저장
            }
            torch.save(save_state, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("--- Training Finished ---")


if __name__ == "__main__":
    config_file_path = "configs/dpr_config.yaml" 
    main(config_file_path)