# G:\DPR-Net\model\dpr_modules.py

import torch
import torch.nn as nn
import open_clip # 'open-clip-torch'
from transformers import AutoTokenizer, MistralForCausalLM, MistralConfig # 'transformers'

# -----------------------------------------------
#  ① CLIP Encoder (Eyes v2) 모듈
# -----------------------------------------------

class DprClipEncoder(nn.Module):
    """
    DPR-Net (V2)를 위한 CLIP 인코더 (Eyes v2)
    
    기능:
    1. [Global] CLS 토큰 임베딩 추출
    2. [Local] Patch 토큰 임베딩 추출
    
    참고: 훈련 중 CLIP의 가중치는 동결(freeze)하는 것을 권장합니다.
    """
    def __init__(self, model_name="ViT-L-14", pretrained="laion2b_s32b_b82k"):
        super().__init__()
        
        print(f"Loading CLIP model: {model_name} ({pretrained})...")
        # CLIP 모델 로드 (HuggingFace Transformers 또는 open_clip 사용 가능)
        try:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        except Exception as e:
            print(f"Error loading OpenCLIP model: {e}")
            print("Attempting to load from HuggingFace Hub (requires 'transformers')...")
            # (대안: open_clip이 안될 경우 HF transformers 사용 로직 - 여기선 open_clip 가정)
            raise e
            
        # CLIP의 visual 모델 (ViT)을 가져옵니다.
        self.visual_encoder = self.clip_model.visual
        
        # 훈련 중에는 CLIP 가중치를 동결
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        print("CLIP model loaded and frozen.")

    def forward(self, x: torch.Tensor):
        """
        이미지 텐서(x)를 입력받아 global 및 patch 임베딩을 반환합니다.
        
        Args:
            x (torch.Tensor): 입력 이미지 텐서 (B, C, H, W). 
                              CLIP이 요구하는 전처리(normalize 등)가 완료된 상태여야 함.
                              
        Returns:
            torch.Tensor: vision_tokens (B, N+1, D) 
                          - [Global, Local_1, Local_2, ...]
            torch.Tensor: global_embed (B, D) 
                          - CLS 토큰 (전역 임베딩)
        """
        
        # ViT의 forward 과정을 수동으로 분해하여 중간 토큰을 가져옵니다.
        # (open_clip의 ViT 기준)
        
        # 1. Patch Embedding
        x = self.visual_encoder.conv1(x)  # (B, D, H', W')
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, D, N_patches)
        x = x.permute(0, 2, 1)  # (B, N_patches, D)
        
        # 2. CLS 토큰 추가 + Positional Embedding
        # (B, 1, D)
        cls_token = self.visual_encoder.class_embedding.expand(x.shape[0], -1, -1) 
        x = torch.cat([cls_token, x], dim=1)  # (B, N_patches + 1, D)
        x = x + self.visual_encoder.positional_embedding
        
        # 3. Transformer 인코더 통과
        x = self.visual_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # (N+1, B, D) - Transformer 입력 형식
        
        # CLIP의 Transformer 레이어 통과
        for layer in self.visual_encoder.transformer.resblocks:
            x = layer(x)
            
        x = x.permute(1, 0, 2)  # (B, N+1, D) - 다시 원래 배치 형식으로
        
        # LayerNorm (final)
        x = self.visual_encoder.ln_post(x)
        
        # 4. 출력 분리
        # vision_tokens: [CLS_token, Patch_token_1, Patch_token_2, ...]
        vision_tokens = x  # (B, N+1, D)
        
        # global_embed: CLS 토큰만 따로 분리
        global_embed = x[:, 0] # (B, D)
        
        return vision_tokens, global_embed


# -----------------------------------------------
#  ② LLM (Brain) 모듈
# -----------------------------------------------

class DprLLM(nn.Module):
    """
    DPR-Net (V2)의 ②번 LLM (Brain) 모듈.
    Mistral 모델을 기반으로 하며, 시각 토큰을 입력받아
    제어용 hidden_state와 설명용 logits를 출력합니다.
    
    Args:
        clip_embed_dim (int): DprClipEncoder의 출력 차원 (예: ViT-L-14 -> 768)
        llm_embed_dim (int): 사용할 LLM의 hidden 차원 (예: Mistral-7B -> 4096)
        model_name (str): HuggingFace의 사전 훈련된 Mistral 모델 이름
    """
    def __init__(self, 
                 clip_embed_dim: int = 768, 
                 llm_embed_dim: int = 4096, 
                 model_name: str = "mistralai/Mistral-7B-v0.1"):
        super().__init__()
        
        print(f"Loading LLM (Brain): {model_name}...")
        
        try:
            # 1. LLM (Mistral) 로드
            self.llm = MistralForCausalLM.from_pretrained(model_name)
            
            # (참고) 텍스트 생성 및 L_consistency를 위한 토크나이저
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading Mistral model: {e}")
            print("Make sure 'transformers' is installed and you have internet access.")
            raise e
            
        # 2. Input Projection Layer
        # CLIP 임베딩 차원(D_clip)을 LLM 임베딩 차원(D_llm)으로 매핑
        self.input_projector = nn.Linear(clip_embed_dim, llm_embed_dim)
        
        # 3. LLM 가중치 동결 (선택적)
        # 제어기(DprLLM)를 훈련할 때, 거대한 LLM 본체는 동결하고
        # input_projector와 FiLM Head만 훈련하는 것이 효율적입니다 (PEFT).
        print("Freezing LLM weights for efficient adapter-style training...")
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # input_projector는 반드시 훈련되어야 합니다.
        for param in self.input_projector.parameters():
            param.requires_grad = True

        print("LLM (Brain) module initialized.")

    def forward(self, vision_tokens: torch.Tensor):
        """
        Args:
            vision_tokens (torch.Tensor): (B, N+1, D_clip) - DprClipEncoder의 출력
                (N+1 = 1 CLS + N Patches)
                
        Returns:
            hidden_state (torch.Tensor): (B, N+1, D_llm) 
                                       - Control Path V2용 (Multi-stage FiLM Head 입력)
            logits (torch.Tensor): (B, N+1, Vocab_Size) 
                                 - Text Path V2용 (설명 생성 입력)
        """
        
        # 1. Project CLIP embeddings to LLM embedding space
        # (B, N+1, D_clip) -> (B, N+1, D_llm)
        llm_embeds = self.input_projector(vision_tokens)
        
        # 2. LLM Forward Pass
        # 시각 임베딩을 텍스트 임베딩처럼 LLM에 입력
        outputs = self.llm(
            inputs_embeds=llm_embeds,
            output_hidden_states=False, # last_hidden_state가 기본 출력임
            return_dict=True
        )
        
        # 3. 출력 추출
        # Control Path용: 마지막 레이어의 모든 토큰 hidden state
        hidden_state = outputs.last_hidden_state # (B, N+1, D_llm)
        
        # Text Path용: 텍스트 생성을 위한 logits
        logits = outputs.logits # (B, N+1, Vocab_Size)
        
        return hidden_state, logits


# -----------------------------------------------
#  ④ Multi-Stage FiLM Head (Controller) 모듈
# -----------------------------------------------

class FiLMHead(nn.Module):
    """
    개별 FiLM 파라미터(gamma, beta) 생성기 (헬퍼 모듈).
    LLM hidden state (D_llm)를 입력받아
    특정 VETNet 스테이지의 채널 차원(stage_dim)에 맞는
    gamma와 beta를 생성합니다.
    """
    def __init__(self, llm_embed_dim: int, stage_dim: int):
        super().__init__()
        # MLP: D_llm -> D_llm/2 -> stage_dim * 2 (for gamma and beta)
        self.projector = nn.Sequential(
            nn.Linear(llm_embed_dim, llm_embed_dim // 2),
            nn.GELU(),
            nn.Linear(llm_embed_dim // 2, stage_dim * 2) # gamma와 beta
        )
    
    def forward(self, x: torch.Tensor):
        # x shape: (B, D_llm)
        gb = self.projector(x) # (B, stage_dim * 2)
        
        # gamma, beta 분리
        gamma, beta = torch.chunk(gb, 2, dim=1) # (B, stage_dim)
        
        # FiLM은 2D Conv 피처 (B, C, H, W)에 적용됩니다.
        # (B, C) -> (B, C, 1, 1)로 reshape 필요
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


class DprMultiStageFiLMHead(nn.Module):
    """
    DPR-Net (V2)의 ④번 Multi-Stage FiLM Head.
    LLM의 토큰별 hidden_state를 입력받아,
    VETNet의 각 스테이지를 제어할 FiLM 신호 딕셔너리를 생성합니다.
    
    V2 아키텍처 (중요):
    - "token_K -> Stage K FiLM"
    - "Global token (CLS) -> Deeper layer 제어" (예: latent)
    - "Patch tokens -> Shallow layer 제어" (예: encoder1)
    """
    def __init__(self, 
                 llm_embed_dim: int = 4096, 
                 vetnet_dim: int = 48):
        super().__init__()
        
        # VETNet (RestormerVolterra)의 스테이지별 차원
        # (restormer_volterra.py의 기본값 기준)
        dim = vetnet_dim
        self.stage_dims = {
            'encoder1':   dim,       # Shallow
            'encoder2':   dim * 2,
            'encoder3':   dim * 4,
            'latent':     dim * 8,   # Deep (Global)
            'decoder3':   dim * 4,
            'decoder2':   dim * 2,
            'decoder1':   dim,       # Shallow
            'refinement': dim,       # Shallow
        }
        self.num_stages = len(self.stage_dims)
        
        print(f"Initializing Multi-Stage FiLM Head for {self.num_stages} VETNet stages.")

        # V2 아키텍처 스펙에 맞춘 토큰-스테이지 매핑
        # (LLM 출력 토큰 [0, 1, 2, 3, 4, 5, 6, 7]을 사용)
        self.token_stage_map = {
            # Stage Name : Token Index
            'encoder1':   1, # 1번 패치 토큰 (Shallow)
            'encoder2':   2, # 2번 패치 토큰
            'encoder3':   3, # 3번 패치 토큰
            'latent':     0, # 0번 CLS (Global) 토큰 (Deep)
            'decoder3':   4, # 4번 패치 토큰
            'decoder2':   5, # 5번 패치 토큰
            'decoder1':   6, # 6번 패치 토큰
            'refinement': 7  # 7번 패치 토큰
        }
        print(f"V2 Stage-Token Map: {self.token_stage_map}")
        
        # 각 스테이지별로 별도의 FiLMHead (MLP) 생성
        self.film_heads = nn.ModuleDict()
        for stage_name, stage_dim in self.stage_dims.items():
            self.film_heads[stage_name] = FiLMHead(llm_embed_dim, stage_dim)

    def forward(self, hidden_state: torch.Tensor):
        """
        Args:
            hidden_state (torch.Tensor): (B, N+1, D_llm) - LLM의 출력
                (N+1 >= 8 이어야 함)
                
        Returns:
            dict: {
                'encoder1': (gamma1, beta1),
                'encoder2': (gamma2, beta2),
                ...
            }
            - 각 (gamma, beta)는 (B, C_stage, 1, 1) 형상
        """
        
        if hidden_state.shape[1] < self.num_stages:
            raise ValueError(f"LLM hidden state has {hidden_state.shape[1]} tokens, "
                             f"but FiLM Head requires at least {self.num_stages} tokens "
                             f"based on the token_stage_map.")
            
        film_signals = {}
        for stage_name, token_index in self.token_stage_map.items():
            # V2 스펙에 따라 지정된 인덱스의 토큰을 선택
            # token shape: (B, D_llm)
            token = hidden_state[:, token_index, :]
            
            # (gamma, beta) 생성
            gamma, beta = self.film_heads[stage_name](token)
            film_signals[stage_name] = (gamma, beta)
            
        return film_signals


# -----------------------------------------------
#  ✅ 테스트 코드 (dpr_modules.py)
# -----------------------------------------------

# (가짜 LLM 설정 클래스 - 빠른 테스트용)
class DummyDprLLM(DprLLM):
    def __init__(self, clip_embed_dim, llm_embed_dim):
        nn.Module.__init__(self) # DprLLM의 __init__ 건너뛰기
        print("Loading DUMMY LLM for fast testing (no download)...")
        
        self.config = MistralConfig(
            hidden_size=llm_embed_dim,
            vocab_size=1000, # small vocab
            num_hidden_layers=2, # very shallow
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=1024
        )
        self.llm = MistralForCausalLM(self.config) # 가짜 모델 생성
        self.tokenizer = None
        self.input_projector = nn.Linear(clip_embed_dim, llm_embed_dim)
        
        # 가중치 동결 (원본과 동일하게)
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.input_projector.parameters():
            param.requires_grad = True
        print("Dummy LLM (Brain) module initialized.")


if __name__ == '__main__':
    
    # 0. 테스트 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # (ViT-L-14 -> Mistral-7B 기반)
    CLIP_DIM = 768
    LLM_DIM = 4096  # Mistral-7B
    VETNET_DIM = 48
    
    # (DprClipEncoder 출력 기준: 1 CLS + 196 Patches)
    NUM_TOKENS = 197 
    BATCH_SIZE = 2

    print(f"--- DPR-Net V2 Module Test (on {device}) ---")
    
    # --------------------------------
    # 1. DprClipEncoder (Eyes) 테스트
    # --------------------------------
    try:
        clip_encoder = DprClipEncoder().to(device)
        clip_encoder.eval() # 추론 모드

        # CLIP이 요구하는 표준 입력 크기 (예: 224x224)
        dummy_image = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)

        print(f"\n[Test 1: DprClipEncoder (Eyes v2)]")
        print(f"Input image shape: {dummy_image.shape}")

        with torch.no_grad():
            vision_tokens, global_embed = clip_encoder(dummy_image)

        print(f"Output vision_tokens shape: {vision_tokens.shape}")
        print(f"Output global_embed shape: {global_embed.shape}")
        
        # (ViT-L-14는 (224/14)^2 = 256개 패치 + CLS 1개 = 257개) -> (수정) L/14는 14x14 패치임. (224/14)^2 = 16*16 = 256. 
        # (ViT-L-14, laion2b_s32b_b82k) -> (224/14)^2 = 256 + 1 = 257 tokens
        # (ViT-B-32) -> (224/32)^2 = 49 + 1 = 50 tokens
        # (ViT-L-14) -> (224/14)^2 = 256 + 1 = 257 tokens
        # (open_clip ViT-L-14 구현은 16x16이 아니라 14x14 패치 크기 사용)
        
        # (수정) 이전 테스트 코드는 ViT-B/16 기준 (197)이었으나, 
        # 기본값 ViT-L/14는 257 토큰 (1 cls + 16*16)이 맞습니다.
        # (수정 2) open_clip ViT-L-14 conv1 (14, 14, stride 14) -> 224/14 = 16. (16*16=256 + 1 = 257)
        # (수정 3) 아, ViT-L-14의 기본값 ViT-L-14@336 (336px)가 아니라 224px 입력 기준입니다.
        # ViT-L-14의 conv1은 kernel_size (14, 14), stride (14, 14) -> 224/14 = 16 -> 16*16 = 256 patches
        
        # (ViT-L-14 @ 224px) -> 1 CLS + (224/14)^2 = 257 tokens
        # (ViT-L-14의 임베딩 차원은 1024)
        # (수정) ViT-L-14는 1024 dim 입니다. ViT-B-14는 768 dim.
        # init()의 기본값이 'ViT-L-14'이므로 1024 dim / 257 tokens가 맞습니다.
        
        # DprClipEncoder 기본값 'ViT-L-14' 기준 (1024 dim)
        CLIP_DIM_L14 = 1024
        NUM_TOKENS_L14 = 257
        
        assert vision_tokens.shape == (BATCH_SIZE, NUM_TOKENS_L14, CLIP_DIM_L14)
        assert global_embed.shape == (BATCH_SIZE, CLIP_DIM_L14)
        
        print(f"✅ DprClipEncoder (Eyes v2) module passed (ViT-L-14).")
        
        # --------------------------------
        # 2. DprLLM (Brain) 테스트
        # --------------------------------
        
        # (테스트용)
        llm_module = DummyDprLLM(CLIP_DIM_L14, LLM_DIM).to(device)
        llm_module.eval()

        # DprClipEncoder의 더미 출력 (B, N+1, D_clip)
        dummy_vision_tokens = torch.randn(BATCH_SIZE, NUM_TOKENS_L14, CLIP_DIM_L14).to(device)
        
        print(f"\n[Test 2: DprLLM (Brain)]")
        print(f"Input vision_tokens shape: {dummy_vision_tokens.shape}")

        with torch.no_grad():
            hidden_state, logits = llm_module(dummy_vision_tokens)
            
        print(f"Output hidden_state shape: {hidden_state.shape}")
        print(f"Output logits shape: {logits.shape}")

        # (B, N+1, D_llm)
        assert hidden_state.shape == (BATCH_SIZE, NUM_TOKENS_L14, LLM_DIM)
        # (B, N+1, Vocab_Size)
        assert logits.shape == (BATCH_SIZE, NUM_TOKENS_L14, llm_module.config.vocab_size)
        print("✅ DprLLM (Brain) module passed.")
        
        # --------------------------------
        # 3. DprMultiStageFiLMHead (Controller) 테스트
        # --------------------------------
        print(f"\n[Test 3: DprMultiStageFiLMHead (Controller)]")
        film_head = DprMultiStageFiLMHead(LLM_DIM, VETNET_DIM).to(device)
        film_head.eval()

        # LLM의 출력을 입력 (NUM_TOKENS_L14 = 257, 8개 이상이므로 통과)
        with torch.no_grad():
            film_signals = film_head(hidden_state) 

        print(f"Output film_signals type: {type(film_signals)}")
        print(f"Total stages controlled: {len(film_signals)}")

        # 8개 스테이지 모두 제어되는지 확인
        assert len(film_signals) == film_head.num_stages

        # 각 스테이지별 (gamma, beta) shape 확인
        for stage_name, (gamma, beta) in film_signals.items():
            expected_dim = film_head.stage_dims[stage_name]
            expected_shape = (BATCH_SIZE, expected_dim, 1, 1)
            
            assert gamma.shape == expected_shape
            assert beta.shape == expected_shape
        
        print(f"Checked shapes for 'latent' (Global): {film_signals['latent'][0].shape}")
        print(f"Checked shapes for 'encoder1' (Local): {film_signals['encoder1'][0].shape}")
        print("✅ DprMultiStageFiLMHead (Controller) module passed.")
        print("\n--- All V2 modules passed! ---")

    except ImportError as e:
        print(f"\n[Error] 'open-clip-torch' or 'transformers' 라이브러리가 필요합니다.")
        print(f"Details: {e}")
        print("pip install open-clip-torch transformers")
    except Exception as e:
        import traceback
        print(f"\n테스트 중 오류 발생: {e}")
        traceback.print_exc()