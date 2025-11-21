# G:\DPR-Net\model\dpr_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
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
#  ② Pixel Decoder Modules (PixelLM 핵심 부품)
# -----------------------------------------------

class PPM(nn.Module):
    """
    Pyramid Pooling Module (PixelLM의 디코더 핵심)
    다양한 스케일(Context)의 정보를 통합합니다.
    """
    def __init__(self, in_dim, reduction_dim, bins):
        super().__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.GELU()
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            # 풀링 후 다시 원래 크기로 Upsample하여 합침
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PixelDecoder(nn.Module):
    """
    LLM의 Hidden State를 받아 이미지 형태의 Feature map으로 변환하는 경량 디코더
    """
    def __init__(self, llm_dim, out_dim):
        super().__init__()
        
        # 1. Projection: LLM dimension -> Decoder dimension (512)
        self.proj = nn.Conv2d(llm_dim, 512, kernel_size=1) 
        
        # 2. PPM (Pyramid Pooling)
        # 512채널 + (128채널 * 4개 bin) = 1024채널
        self.ppm = PPM(512, int(512/4), [1, 2, 3, 6])
        
        # 3. Final Fusion
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv2d(512, out_dim, kernel_size=1) # 최종 출력 (FiLM Head 입력 차원과 맞춤)
        )

    def forward(self, llm_hidden, h, w):
        """
        llm_hidden: (B, N_patches, D) - CLS 토큰 제외됨
        h, w: 원래 비전 토큰의 그리드 크기 (예: 16x16)
        """
        # (B, N, D) -> (B, D, N)
        x = llm_hidden.permute(0, 2, 1).contiguous() 
        # (B, D, H, W) 형태로 변환 (Spatial Reshape)
        x = x.view(x.size(0), x.size(1), h, w) 
        
        # Pixel Reasoning
        x = self.proj(x)
        x = self.ppm(x)
        x = self.bottleneck(x)
        
        return x # (B, out_dim, H, W)


# -----------------------------------------------
#  ③ DprPixelLM (Brain + Decoder) [기존 DprLLM 대체]
# -----------------------------------------------

class DprPixelLM(nn.Module):
    """
    Mistral + Pixel Decoder를 결합한 PixelLM 아키텍처 구현체.
    LLM의 추론 결과를 픽셀 레벨로 디코딩하여 FiLM Head에 전달합니다.
    """
    def __init__(self, 
                 clip_embed_dim: int = 1024, # ViT-L-14 output
                 llm_embed_dim: int = 4096, 
                 model_name: str = "mistralai/Mistral-7B-v0.1"):
        super().__init__()
        
        print(f"Loading PixelLM Core (Based on {model_name})...")
        
        try:
            # 1. LLM (Mistral) 로드 (8-bit 양자화)
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.llm = MistralForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
            # --- LoRA 어댑터 적용 ---
            lora_config = LoraConfig(
                r=16, # PixelLM은 표현력이 중요하므로 Rank 상향 권장 (8->16)
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 모든 어텐션 레이어 타겟
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, lora_config)
            print("Applied LoRA to PixelLM Brain.")
            self.llm.print_trainable_parameters()

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            print(f"Error loading Mistral model: {e}")
            raise e
        
        # 2. Input Projector (Eyes -> Brain)
        self.input_projector = nn.Linear(clip_embed_dim, llm_embed_dim)
        
        # 3. ★ [NEW] Lightweight Pixel Decoder ★
        # LLM의 출력을 다시 시각적 특징맵으로 정제
        # 출력 차원을 llm_embed_dim(4096)으로 맞추어 기존 FiLM Head와 호환성 유지
        self.pixel_decoder = PixelDecoder(llm_dim=llm_embed_dim, out_dim=llm_embed_dim)
        
        # 4. Trainable 설정
        for param in self.input_projector.parameters(): param.requires_grad = True
        for param in self.pixel_decoder.parameters(): param.requires_grad = True # 디코더 학습 필수
        
        print("✅ DprPixelLM initialized (LLM + Pixel Decoder).")

    def forward(self, vision_tokens, text_labels=None, text_attention_mask=None):
        """
        vision_tokens: CLIP에서 나온 (B, 257, D) 형태. [CLS, Patch_1, ..., Patch_256]
        """
        # 1. Vision Embedding Projection
        vision_embeds = self.input_projector(vision_tokens) # (B, 257, 4096)
        
        loss_text = None
        logits = None
        generated_ids = None

        # 2. LLM Input 구성
        if text_labels is not None:
            # --- Train Mode: Vision + Text ---
            text_embeds = self.llm.get_input_embeddings()(text_labels)
            inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
            
            # Attention Mask
            vision_mask = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=vision_embeds.device)
            full_attention_mask = torch.cat([vision_mask, text_attention_mask], dim=1)
            
            # Labels (Vision 부분은 -100으로 마스킹)
            vision_labels = torch.full_like(vision_mask, -100)
            full_labels = torch.cat([vision_labels, text_labels], dim=1)

            # LLM Forward
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels,
                output_hidden_states=True,
                return_dict=True
            )
            loss_text = outputs.loss
            logits = outputs.logits
            
            # Hidden State 추출 (전체 시퀀스 중 Vision Token 부분만 필요함)
            # outputs.hidden_states[-1] shape: (B, 257 + Text_Len, D)
            total_hidden = outputs.hidden_states[-1]
            vision_seq_len = vision_tokens.shape[1] # 257
            vision_hidden_state = total_hidden[:, :vision_seq_len, :] # (B, 257, D)
            
        else:
            # --- Eval Mode: Vision Only ---
            inputs_embeds = vision_embeds
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=True
            )
            vision_hidden_state = outputs.hidden_states[-1] # (B, 257, D)
            
            # 텍스트 생성 (선택 사항)
            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=200,
                num_beams=2,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 3. ★ Pixel Reasoning (Decoding) ★
        # LLM이 이해한 정보를 Pixel Decoder에 통과시켜 공간 정보를 복원합니다.
        
        # CLS 토큰과 Patch 토큰 분리
        cls_token = vision_hidden_state[:, 0:1, :] # (B, 1, D) - Global Semantic
        patch_tokens = vision_hidden_state[:, 1:, :] # (B, 256, D) - Local Patches
        
        # Reshape: (B, 256, D) -> Grid (16x16)
        # ViT-L/14는 224x224 이미지를 14x14 패치로 쪼갬 -> 16x16 Grid (256개)
        # (만약 224 이미지라면 16x16=256)
        grid_size = int(patch_tokens.shape[1] ** 0.5) # 16
        
        # Pixel Decoder 통과 -> (B, D, H, W)
        decoded_map = self.pixel_decoder(patch_tokens, h=grid_size, w=grid_size)
        
        # FiLM Head를 위해 다시 시퀀스로 변환
        # (B, D, H, W) -> (B, D, H*W) -> (B, N, D)
        decoded_seq = decoded_map.flatten(2).transpose(1, 2) # (B, 256, D)
        
        # 최종 결합: CLS(LLM Pure) + Patches(Pixel Decoded)
        final_features = torch.cat([cls_token, decoded_seq], dim=1) # (B, 257, D)
        
        if text_labels is not None:
            return loss_text, final_features, logits
        else:
            return generated_ids, final_features, None


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

if __name__ == '__main__':
    
    print("--- DPR-Net PixelLM Module Test ---")
    
    try:
        # 0. 테스트 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        CLIP_DIM_L14 = 1024
        LLM_DIM = 4096 
        VETNET_DIM = 48
        NUM_TOKENS_L14 = 257 # 1 CLS + 256 Patches
        BATCH_SIZE = 2
        
        # 1. DprClipEncoder (Eyes) 테스트
        clip_encoder = DprClipEncoder().to(device)
        clip_encoder.eval() 
        dummy_image = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
        print(f"\n[Test 1: DprClipEncoder]")
        with torch.no_grad():
            vision_tokens, _ = clip_encoder(dummy_image)
        assert vision_tokens.shape == (BATCH_SIZE, NUM_TOKENS_L14, CLIP_DIM_L14)
        print(f"✅ DprClipEncoder passed.")
        
        # 2. PixelDecoder Unit Test
        print(f"\n[Test 2: PixelDecoder]")
        decoder = PixelDecoder(llm_dim=LLM_DIM, out_dim=LLM_DIM).to(device)
        dummy_llm_out = torch.randn(BATCH_SIZE, 256, LLM_DIM).to(device) # 16x16 patches
        decoded = decoder(dummy_llm_out, 16, 16)
        assert decoded.shape == (BATCH_SIZE, LLM_DIM, 16, 16)
        print(f"✅ PixelDecoder passed.")

        print("\n[Note] DprPixelLM requires real Mistral weights. Skipping full integration test in this script.")
        print("Please run dpr_net.py or train.py to load actual models.")

    except ImportError as e:
        print(f"\n[Error] Dependencies missing: {e}")
    except Exception as e:
        import traceback
        print(f"\nTest Error: {e}")
        traceback.print_exc()