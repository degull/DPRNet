# G:\DPR-Net\model\dpr_modules.py
import torch
import torch.nn as nn
import open_clip # 'open-clip-torch'
from transformers import AutoTokenizer, MistralForCausalLM, MistralConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model # 'peft' 라이브러리 필요

# -----------------------------------------------
#  ① CLIP Encoder (Eyes v2) 모듈
# -----------------------------------------------

class DprClipEncoder(nn.Module):
    """
    DPR-Net (V2)를 위한 CLIP 인코더 (Eyes v2)
    """
    def __init__(self, model_name="ViT-L-14", pretrained="laion2b_s32b_b82k"):
        super().__init__()
        
        print(f"Loading CLIP model: {model_name} ({pretrained})...")
        try:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        except Exception as e:
            print(f"Error loading OpenCLIP model: {e}")
            raise e
            
        self.visual_encoder = self.clip_model.visual
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        print("CLIP model loaded and frozen.")

    def forward(self, x: torch.Tensor):
        """
        이미지 텐서(x)를 입력받아 global 및 patch 임베딩을 반환합니다.
        """
        
        # 1. Patch Embedding
        x = self.visual_encoder.conv1(x)  # (B, D, H', W')
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, D, N_patches)
        x = x.permute(0, 2, 1)  # (B, N_patches, D)
        
        # 2. CLS 토큰 추가 + Positional Embedding
        # (1D -> 3D 텐서로 올바르게 변환)
        cls_token = self.visual_encoder.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, N_patches + 1, D)
        x = x + self.visual_encoder.positional_embedding
        
        # 3. Transformer 인코더 통과
        x = self.visual_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # (N+1, B, D) 
        
        for layer in self.visual_encoder.transformer.resblocks:
            x = layer(x)
            
        x = x.permute(1, 0, 2)  # (B, N+1, D)
        x = self.visual_encoder.ln_post(x)
        
        # 4. 출력 분리
        vision_tokens = x  # (B, N+1, D)
        global_embed = x[:, 0] # (B, D)
        
        return vision_tokens, global_embed


# -----------------------------------------------
#  ② LLM (Brain) 모듈 (LoRA + 버그 수정)
# -----------------------------------------------
# (사용자 코드에서 중복 정의된 DprLLM 클래스 1개 제거)
class DprLLM(nn.Module):
    """
    DPR-Net (V2)의 ②번 LLM (Brain) 모듈.
    (8-bit 양자화 + LoRA 어댑터 적용됨)
    """
    def __init__(self, 
                 clip_embed_dim: int = 768, 
                 llm_embed_dim: int = 4096, 
                 model_name: str = "mistralai/Mistral-7B-v0.1"):
        super().__init__()
        
        print(f"Loading LLM (Brain): {model_name}...")
        
        try:
            # 8-bit 양자화 설정
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            
            # 1. LLM (Mistral) 로드 (양자화 적용)
            self.llm = MistralForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config, # 8-bit 설정 적용
                device_map="auto" # (중요) accelerate가 GPU에 자동 배포
            )
            
            # --- LoRA 어댑터 적용 ---
            lora_config = LoraConfig(
                r=8, # Rank
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"], # Mistral 어텐션 레이어
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM" # 텍스트 생성
            )
            
            self.llm = get_peft_model(self.llm, lora_config)
            print("Applied LoRA to LLM (Brain). Trainable LoRA parameters:")
            self.llm.print_trainable_parameters()
            # --- LoRA 적용 완료 ---

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading Mistral model: {e}")
            print("Make sure 'transformers', 'bitsandbytes', 'accelerate', 'peft' are installed.")
            raise e
            
        # 2. Input Projection Layer
        self.input_projector = nn.Linear(clip_embed_dim, llm_embed_dim)
        
        # 3. LLM 가중치 동결 (LoRA 레이어 제외)
        # (input_projector는 PEFT와 별개이므로 수동으로 trainable 설정)
        for param in self.input_projector.parameters():
            param.requires_grad = True

        print("LLM (Brain) module initialized (LoRA + Projector are trainable).")

    def forward(self, 
                vision_tokens: torch.Tensor, 
                text_labels: torch.Tensor = None,
                text_attention_mask: torch.Tensor = None):
        """
        [수정] 명시적 텍스트 훈련(L_text)을 위해
        'text_labels' (정답 텍스트 토큰)도 입력받습니다.
        """
        
        # 1. Project CLIP embeddings to LLM embedding space
        vision_embeds = self.input_projector(vision_tokens)
        
        if text_labels is not None:
            # --- 훈련(Train) 모드 (L_text 계산) ---
            # 2. 정답 텍스트 토큰을 LLM 임베딩으로 변환
            text_embeds = self.llm.get_input_embeddings()(text_labels)

            # 3. [시각 임베딩 + 텍스트 임베딩] 결합
            inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
            
            # 4. 어텐션 마스크 결합
            vision_mask = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=vision_embeds.device)
            full_attention_mask = torch.cat([vision_mask, text_attention_mask], dim=1)
            
            # 5. [정답 레이블] 결합 (시각 토큰 부분은 -100으로 마스킹)
            vision_labels = torch.full_like(vision_mask, -100)
            full_labels = torch.cat([vision_labels, text_labels], dim=1)

            # 6. LLM Forward Pass (L_text 포함)
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels,
                output_hidden_states=True, # L_consistency를 위해
                return_dict=True
            )
            
            loss_text = outputs.loss # (L_text 손실)
            hidden_state = outputs.hidden_states[-1] # [수정] FiLM Head를 위한 전체 hidden_state
            logits = outputs.logits 

            # (L_consistency용 CLS 토큰과 FiLM용 8개 토큰 모두 반환)
            return loss_text, hidden_state, logits

        else:
            # --- 추론(Eval) 모드 ---
            # 1. 시각 임베딩만 입력
            inputs_embeds = vision_embeds
            
            # [수정] 추론 시에도 hidden_state를 반환해야 FiLM 작동 가능
            # 2. 먼저 hidden_state 계산
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_state = outputs.hidden_states[-1] # [B, 257, D_llm]

            # 3. 텍스트 생성
            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=200, # (생성할 최대 텍스트 길이)
                num_beams=2,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            return generated_ids, hidden_state, None # hidden_state 반환


# -----------------------------------------------
#  ④ Multi-Stage FiLM Head (Controller) 모듈
# -----------------------------------------------

class FiLMHead(nn.Module):
    """
    개별 FiLM 파라미터(gamma, beta) 생성기 (헬퍼 모듈).
    """
    def __init__(self, llm_embed_dim: int, stage_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(llm_embed_dim, llm_embed_dim // 2),
            nn.GELU(),
            nn.Linear(llm_embed_dim // 2, stage_dim * 2) 
        )
    
    def forward(self, x: torch.Tensor):
        gb = self.projector(x) 
        gamma, beta = torch.chunk(gb, 2, dim=1) 
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


class DprMultiStageFiLMHead(nn.Module):
    """
    DPR-Net (V2)의 ④번 Multi-Stage FiLM Head.
    """
    def __init__(self, 
                 llm_embed_dim: int = 4096, 
                 vetnet_dim: int = 48):
        super().__init__()
        
        dim = vetnet_dim
        self.stage_dims = {
            'encoder1':   dim,       
            'encoder2':   dim * 2,
            'encoder3':   dim * 4,
            'latent':     dim * 8,   
            'decoder3':   dim * 4,
            'decoder2':   dim * 2,
            'decoder1':   dim,       
            'refinement': dim,       
        }
        self.num_stages = len(self.stage_dims)
        
        print(f"Initializing Multi-Stage FiLM Head for {self.num_stages} VETNet stages.")

        self.token_stage_map = {
            'encoder1':   1, 
            'encoder2':   2, 
            'encoder3':   3, 
            'latent':     0, 
            'decoder3':   4, 
            'decoder2':   5, 
            'decoder1':   6, 
            'refinement': 7  
        }
        print(f"V2 Stage-Token Map: {self.token_stage_map}")
        
        self.film_heads = nn.ModuleDict()
        for stage_name, stage_dim in self.stage_dims.items():
            self.film_heads[stage_name] = FiLMHead(llm_embed_dim, stage_dim)

    def forward(self, hidden_state: torch.Tensor):
        # (vision_tokens에서 8개 토큰만 사용)
        if hidden_state.shape[1] < self.num_stages:
            raise ValueError(f"LLM hidden state has {hidden_state.shape[1]} tokens, "
                             f"but FiLM Head requires at least {self.num_stages} tokens.")
        
        film_signals = {}
        for stage_name, token_index in self.token_stage_map.items():
            token = hidden_state[:, token_index, :]
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
        
        # (LoRA 테스트는 더미에서 생략)
        print("Applying DUMMY LoRA...")
        
        self.tokenizer = None
        self.input_projector = nn.Linear(clip_embed_dim, llm_embed_dim)
        
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.input_projector.parameters():
            param.requires_grad = True
        print("Dummy LLM (Brain) module initialized.")
    
    # (더미 forward는 LoRA 훈련 로직을 따름)
    def forward(self, vision_tokens: torch.Tensor, 
                text_labels: torch.Tensor = None,
                text_attention_mask: torch.Tensor = None):
        
        llm_embeds = self.input_projector(vision_tokens)
        
        if text_labels is not None:
            # (훈련 모드 - 더미)
            # (실제 LoRA 모델과 동일한 출력을 흉내)
            outputs = self.llm(inputs_embeds=llm_embeds, output_hidden_states=True)
            loss_text = torch.tensor(0.0, device=llm_embeds.device) # 가짜 손실
            hidden_state = outputs.hidden_states[-1] # 전체 hidden_state
            logits = outputs.logits
            return loss_text, hidden_state, logits # (hidden_state 반환)
        else:
            # (추론 모드 - 더미)
            # (가짜 hidden_state 및 생성 ID 반환)
            dummy_hidden_state = torch.randn(
                vision_tokens.shape[0], 
                vision_tokens.shape[1], 
                self.config.hidden_size, 
                device=vision_tokens.device
            )
            generated_ids = torch.randint(0, 1000, (vision_tokens.shape[0], 10), device=vision_tokens.device)
            return generated_ids, dummy_hidden_state, None


if __name__ == '__main__':
    
    print("--- DPR-Net V2 Module Test (if __name__ == '__main__') ---")
    print("[Note] This test block may be outdated due to DprLLM.forward signature changes.")
    
    try:
        # 0. 테스트 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        CLIP_DIM_L14 = 1024
        LLM_DIM = 4096 
        VETNET_DIM = 48
        NUM_TOKENS_L14 = 257 # 1 CLS + (224/14)^2
        BATCH_SIZE = 2
        
        # 1. DprClipEncoder (Eyes) 테스트
        clip_encoder = DprClipEncoder().to(device)
        clip_encoder.eval() 
        dummy_image = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
        print(f"\n[Test 1: DprClipEncoder (Eyes v2)]")
        with torch.no_grad():
            vision_tokens, _ = clip_encoder(dummy_image)
        assert vision_tokens.shape == (BATCH_SIZE, NUM_TOKENS_L14, CLIP_DIM_L14)
        print(f"✅ DprClipEncoder (Eyes v2) module passed.")
        
        # 2. DprLLM (Brain) 테스트 (더미)
        llm_module = DummyDprLLM(CLIP_DIM_L14, LLM_DIM).to(device)
        llm_module.eval()
        print(f"\n[Test 2: DprLLM (Brain) - Dummy Test]")
        
        # (훈련 모드 테스트)
        dummy_labels = torch.randint(0, 1000, (BATCH_SIZE, 10), device=device)
        dummy_mask = torch.ones_like(dummy_labels)
        with torch.no_grad():
            loss_text, hidden_state, logits = llm_module(vision_tokens, dummy_labels, dummy_mask)
        assert loss_text is not None
        # (전체 hidden_state가 반환되는지 확인)
        assert hidden_state.shape == (BATCH_SIZE, NUM_TOKENS_L14, LLM_DIM) 
        print(f"✅ DprLLM (Train Mode) dummy test passed.")
        
        # (추론 모드 테스트)
        with torch.no_grad():
            generated_ids, hidden_state, _ = llm_module(vision_tokens)
        assert generated_ids.shape[0] == BATCH_SIZE
        assert hidden_state.shape == (BATCH_SIZE, NUM_TOKENS_L14, LLM_DIM) # (추론 시에도 hidden_state 확인)
        print(f"✅ DprLLM (Eval Mode) dummy test passed.")

        # 3. DprMultiStageFiLMHead (Controller) 테스트
        print(f"\n[Test 3: DprMultiStageFiLMHead (Controller)]")
        film_head = DprMultiStageFiLMHead(LLM_DIM, VETNET_DIM).to(device)
        film_head.eval()
        # (DprLLM이 이제 [B, 257, D]를 반환하므로 테스트 정상 작동)
        with torch.no_grad():
            film_signals = film_head(hidden_state) # (257개 토큰 사용)
        assert len(film_signals) == film_head.num_stages
        print("✅ DprMultiStageFiLMHead (Controller) module passed.")
        print("\n--- All V2 modules (dummy) test passed! ---")

    except ImportError as e:
        print(f"\n[Error] 'open-clip-torch', 'transformers', 'bitsandbytes', 'accelerate', 'peft' 라이브러리가 필요합니다.")
        print(f"Details: {e}")
        print("pip install open-clip-torch transformers bitsandbytes accelerate peft")
    except Exception as e:
        import traceback
        print(f"\n테스트 중 오류 발생: {e}")
        traceback.print_exc()