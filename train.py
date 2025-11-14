# G:\DPR-Net\train.py

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import open_clip # 'open-clip-torch'

# --- 1. 모델 임포트 ---
# (G:\DPR-Net\model\ 폴더에서 DPR_Net을 불러옵니다)
# (테스트를 위해 DummyDPR_Net도 임포트합니다)
try:
    from model.dpr_net import DPR_Net, DummyDPR_Net
except ImportError:
    print("Error: 'model' 패키지를 찾을 수 없습니다.")
    print("G:\\DPR-Net\\model\\__init__.py 파일이 있는지 확인하세요.")
    exit(1)

# -----------------------------------------------
#  ⑦ V2 Training Loss (L_total)
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
                 llm_embed_dim=4096, # Mistral-7B
                 clip_embed_dim=1024 # ViT-L-14
                 ):
        super().__init__()
        print(f"Initializing DPRV2Loss (λ_film={lambda_film}, λ_consistency={lambda_consistency})")
        
        self.lambda_film = lambda_film
        self.lambda_consistency = lambda_consistency

        # --- L_img (복원 손실) ---
        # L1 Loss가 이미지 복원에서 가장 보편적으로 사용됩니다.
        self.l1_loss = nn.L1Loss()
        
        # --- L_consistency (일관성 손실) ---
        # "|| CLIP_text_embed(text) - hidden_state ||^2"
        
        # 1. MSE (L2) 손실
        self.mse_loss = nn.MSELoss()
        
        # 2. CLIP Text Encoder (L_consistency 계산용)
        print("  Loading CLIP Text Encoder for L_consistency...")
        self.clip_text_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        
        # CLIP 텍스트 인코더는 동결 (학습 대상 아님)
        for param in self.clip_text_model.parameters():
            param.requires_grad = False
        print("  CLIP Text Encoder frozen.")
            
        # 3. Projector (D_llm -> D_clip)
        # LLM hidden_state 차원(4096)을 CLIP 텍스트 임베딩 차원(1024)으로 매핑
        self.consistency_projector = nn.Linear(llm_embed_dim, clip_embed_dim)
        
        # (참고) L_consistency를 위한 projector는 DPRV2Loss 모듈이 소유하며
        # 메인 optimizer에 의해 학습되어야 합니다.

    def calculate_film_loss(self, hidden_state):
        """
        L_film (Stage Alignment) 계산.
        V2 스펙: 각 스테이지를 제어하는 토큰(0~7)들이
        서로 다른 역할을 하도록 강제 (Decorrelation)
        """
        # (B, N+1, D_llm)
        num_tokens_to_align = 8 # (token_stage_map 기준 0~7)
        if hidden_state.shape[1] < num_tokens_to_align:
            return 0 # 계산 불가
            
        # (B, 8, D_llm)
        control_tokens = hidden_state[:, :num_tokens_to_align, :]
        
        # L2 정규화
        tokens_norm = F.normalize(control_tokens, p=2, dim=2)
        
        # (B, 8, D_llm) @ (B, D_llm, 8) -> (B, 8, 8)
        # 코사인 유사도 행렬
        cosine_sim_matrix = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))
        
        # 우리는 '비'대각(off-diagonal) 요소 (토큰 간 유사도)가
        # 0이 되기를 원합니다. (대각 요소는 항상 1)
        
        # (B, 8, 8)에서 대각 요소를 0으로 만듦
        eye = torch.eye(num_tokens_to_align, device=cosine_sim_matrix.device).expand_as(cosine_sim_matrix)
        off_diagonal_sim = cosine_sim_matrix * (1 - eye)
        
        # L_film: 비대각 요소들의 제곱 평균 (0에 가까워지도록)
        loss_film = torch.mean(off_diagonal_sim.pow(2))
        return loss_film

    def calculate_consistency_loss(self, hidden_state, logits, llm_tokenizer, device):
        """
        L_consistency (설명-제어 일치) 계산.
        V2 스펙: || Project(LLM_Global_Token) - CLIP_Embed(Generated_Text) ||^2
        """
        
        # --- 1. Control Path: LLM Global Token (제어 신호) ---
        # (B, N+1, D_llm) -> (B, D_llm)
        llm_global_hidden = hidden_state[:, 0, :] # CLS 토큰
        
        # (B, D_llm) -> (B, D_clip)
        projected_hidden = self.consistency_projector(llm_global_hidden)
        
        
        # --- 2. Text Path: CLIP Embed(Generated_Text) (설명 신호) ---
        
        # (B, N+1, Vocab) -> (B, N+1)
        # Logits에서 텍스트 토큰 생성 (Greedy decoding)
        # (참고: 이 연산은 .detach()로 인해 역전파 그래프에 포함되지 않음)
        generated_tokens = torch.argmax(logits.detach(), dim=-1)
        
        # (B, N+1) -> [str_b1, str_b2, ...]
        # 생성된 토큰 ID를 실제 텍스트 문자열로 디코딩
        generated_text_list = llm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # [str] -> (B, 77)
        # 디코딩된 텍스트를 CLIP 토크나이저로 다시 토큰화
        clip_text_tokens = self.clip_tokenizer(generated_text_list).to(device)
        
        # (B, 77) -> (B, D_clip)
        # CLIP 텍스트 인코더로 임베딩
        with torch.no_grad(): # CLIP 인코더는 항상 동결
            clip_text_embed = self.clip_text_model.encode_text(clip_text_tokens)
            
        # (중요) 타겟 임베딩은 역전파가 흐르지 않도록 detach()
        target_text_embed = clip_text_embed.float().detach()

        
        # --- 3. L2 Loss (MSE) ---
        # || 제어 신호 - 설명 신호 ||^2
        loss_consistency = self.mse_loss(projected_hidden, target_text_embed)
        
        return loss_consistency

    def forward(self, outputs, target_img, llm_tokenizer, device):
        """
        Args:
            outputs (dict): DPR_Net의 forward 출력 딕셔너리
            target_img (torch.Tensor): (B, 3, H, W) 정답(clean) 이미지
            llm_tokenizer: model.tokenizer (텍스트 생성용)
            device: 현재 장치
            
        Returns:
            total_loss (torch.Tensor): L_total (역전파용)
            loss_dict (dict): 로깅용 개별 손실
        """
        
        # 1. L_img (복원 손실)
        l_img = self.l1_loss(outputs['img_restored'], target_img)
        
        # 2. L_film (스테이지 정렬 손실)
        l_film = self.calculate_film_loss(outputs['hidden_state'])
        
        # 3. L_consistency (일관성 손실)
        l_consistency = self.calculate_consistency_loss(
            outputs['hidden_state'], 
            outputs['logits'], 
            llm_tokenizer,
            device
        )
        
        # 4. L_total
        total_loss = l_img + \
                     self.lambda_film * l_film + \
                     self.lambda_consistency * l_consistency
                     
        loss_dict = {
            'total': total_loss.item(),
            'img': l_img.item(),
            'film': l_film.item(),
            'consistency': l_consistency.item()
        }
        
        return total_loss, loss_dict


# -----------------------------------------------
#  (임시) 더미 데이터셋 (G:\DPR-Net\data\dataset.py 대체)
# -----------------------------------------------
class DummyImageDataset(Dataset):
    """ 테스트를 위한 임의의 (왜곡, 정답) 이미지 쌍 생성 """
    def __init__(self, num_samples=1000, img_size=(128, 128)):
        self.num_samples = num_samples
        self.img_size = img_size
        print(f"Using DummyImageDataset with {num_samples} random samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (0~1 범위의 랜덤 텐서)
        h, w = self.img_size
        distorted = torch.rand(3, h, w)
        clean = torch.rand(3, h, w)
        return distorted, clean


# -----------------------------------------------
#  🚀 메인 학습 스크립트 (train.py)
# -----------------------------------------------
def main(args):
    # --- 1. 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 체크포인트 저장 경로
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- 2. 데이터 로더 ---
    # (TODO: G:\DPR-Net\data\dataset.py의 실제 데이터셋으로 교체 필요)
    train_dataset = DummyImageDataset(num_samples=100, img_size=(args.patch_size, args.patch_size))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # (TODO: Validation loader 추가)

    # --- 3. 모델 초기화 ---
    # (실제 학습 시)
    # model = DPR_Net(vetnet_dim=48, ...).to(device)
    
    # (빠른 테스트용: Mistral-7B 다운로드 X)
    print("Initializing DPR-Net in [Dummy Mode] for training test...")
    model_config = {
        'vetnet_dim': 48,
        'vetnet_num_blocks': [2, 2, 2, 2], # (테스트용 축소)
        'vetnet_refinement_blocks': 2,
        'clip_embed_dim': 1024, # ViT-L-14
        'llm_embed_dim': 4096,  # Mistral-7B
    }
    model = DummyDPR_Net(**model_config).to(device)
    
    # --- 4. 손실 함수 초기화 ---
    criterion = DPRV2Loss(
        lambda_film=args.lambda_film,
        lambda_consistency=args.lambda_consistency,
        llm_embed_dim=model_config['llm_embed_dim'],
        clip_embed_dim=model_config['clip_embed_dim']
    ).to(device)

    # --- 5. 옵티마이저 설정 ---
    # V2 아키텍처: LLM/CLIP 본체는 '동결' (frozen)
    # 학습 대상: VETNet, FiLMHead, LLM_Input_Projector, Consistency_Projector
    
    # 1. DPR_Net 내부의 학습 가능한 파라미터
    # (dpr_modules.py에서 requires_grad=True로 설정한 레이어들 + VETNet 전체)
    params_to_train = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # 2. DPRV2Loss 내부의 학습 가능한 파라미터 (consistency_projector)
    params_to_train.extend(list(criterion.consistency_projector.parameters()))
    
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in params_to_train):,}")
    
    optimizer = AdamW(params_to_train, lr=args.learning_rate, weight_decay=1e-4)
    
    # (TODO: Learning rate scheduler 추가)

    # --- 6. 학습 루프 ---
    print(f"\n--- Starting DPR-Net V2 Training ---")
    for epoch in range(args.epochs):
        model.train() # 학습 모드
        criterion.consistency_projector.train() # 손실 함수 내 프로젝터도 학습 모드
        
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for distorted_img, clean_img in pbar:
            distorted_img = distorted_img.to(device)
            clean_img = clean_img.to(device)
            
            # 1. Forward pass
            outputs = model(distorted_img)
            
            # 2. Calculate V2 Loss
            loss, loss_dict = criterion(
                outputs, 
                clean_img, 
                model.tokenizer, # (DprLLM이 소유한) LLM 토크나이저 전달
                device
            )
            
            # 3. Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(
                total=f"{loss_dict['total']:.4f}",
                img=f"{loss_dict['img']:.4f}",
                film=f"{loss_dict['film']:.4f}",
                cons=f"{loss_dict['consistency']:.4f}"
            )
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}")
        
        # (TODO: Validation step)

        # --- 7. 체크포인트 저장 ---
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"dpr_net_epoch_{epoch+1}.pth")
            
            # (모델 + 손실함수 프로젝터 + 옵티마이저 저장)
            save_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(save_state, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("--- Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPR-Net (V2)")
    
    # 경로
    parser.add_argument("--data_dir", type=str, default="./data/train", help="Training data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=128, help="Training patch size (H and W)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    # V2 손실 람다 (가중치)
    parser.add_argument("--lambda_film", type=float, default=0.1, help="Weight for L_film (stage alignment)")
    parser.add_argument("--lambda_consistency", type=float, default=0.01, help="Weight for L_consistency (control-text)")
    
    # 기타
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")

    args = parser.parse_args()
    main(args)