# 뇌 (Mistral + LoRA + Text Generation)
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# ==============================================================================
# 🧠 The Brain: Mistral-7B LLM with LoRA
# 역할: 시각적 토큰과 텍스트 토큰을 결합하여 '멀티모달 추론'을 수행합니다.
# 핵심: Projector(차원 변환) + Concatenation + LoRA + Gradient Stabilization
# ==============================================================================

class MistralLLM(nn.Module):
    def __init__(self, model_id="mistralai/Mistral-7B-v0.1", vision_hidden_size=1024, llm_hidden_size=4096):
        """
        Args:
            model_id: Mistral 모델 ID
            vision_hidden_size: CLIP 출력 차원 (기본 1024)
            llm_hidden_size: Mistral 입력 차원 (기본 4096)
        """
        super().__init__()
        print(f"🧠 Loading Mistral LLM: {model_id}...")
        
        # 1. Tokenizer 로드 (패딩 토큰 설정 필수)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token # Mistral은 pad_token이 없으므로 eos로 대체
        
        # 2. Mistral 모델 로드 (FP16/BF16 권장 - 메모리 절약)
        # device_map="auto"를 통해 가능한 경우 GPU에 자동 할당
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # 3. Projector (The Adapter)
        # CLIP의 1024차원 벡터를 Mistral의 4096차원 공간으로 쏘아올리는 선형 변환
        self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)
        
        # 4. LoRA (Low-Rank Adaptation) 설정
        # 거대 모델 전체를 학습하는 건 불가능하므로, 일부 파라미터(LoRA Adapter)만 학습
        print("🚀 Applying LoRA (Low-Rank Adaptation)...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            r=8,            # Rank (높을수록 표현력 증가, 메모리 증가)
            lora_alpha=32,  # Scaling Factor
            lora_dropout=0.05,
            # Mistral의 모든 Attention Linear Layer 타겟팅
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters() # 학습 가능한 파라미터 비율 출력

    def forward(self, image_embeds, input_ids, attention_mask):
        """
        Args:
            image_embeds: [Batch, 257, 1024] (from CLIP)
            input_ids: [Batch, Text_Len] (Text Tokens)
            attention_mask: [Batch, 257 + Text_Len] (Combined Mask)
            
        Returns:
            last_hidden_state: [Batch, 257 + Text_Len, 4096]
        """
        # 1. Vision Embedding Projection
        # [B, 257, 1024] -> [B, 257, 4096]
        image_embeds_proj = self.projector(image_embeds)
        
        # 2. Text Embedding Extraction
        # Mistral 내부의 Embedding Layer를 사용해 텍스트 ID를 벡터로 변환
        # [B, Text_Len] -> [B, Text_Len, 4096]
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 3. Concatenation (이어 붙이기)
        # [Vision Tokens] + [Text Tokens]
        inputs_embeds = torch.cat([image_embeds_proj, text_embeds], dim=1)
        
        # ⚡ 4. Gradient Checkpointing Stabilization (명세서 핵심)
        # 이 텐서에 grad 추적을 강제하여, Projector와 LLM 사이의 Backprop 단절을 방지합니다.
        # (특히 LLM 본체가 Frozen 상태이거나 LoRA를 쓸 때 입력단 Grad가 끊기는 문제 해결)
        if inputs_embeds.requires_grad:
            inputs_embeds.retain_grad()
        
        # 5. Mistral Forward Pass
        # input_ids 대신 inputs_embeds를 직접 넣어줍니다.
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True # Hidden State 반환 활성화
        )
        
        # 전체 Hidden State 반환
        # (Slicing은 Pixel Decoder 혹은 Main Model에서 수행)
        return outputs.hidden_states[-1]

    # ==========================================================================
    # 🗣️ Optional: Text Generation Mode (for Debugging/Captioning)
    # 명세서의 "설명 모드" 구현을 위한 함수
    # ==========================================================================
    def generate_caption(self, image_embeds, input_ids, max_new_tokens=50):
        """
        이미지를 보고 LLM이 생각하는 내용을 텍스트로 출력합니다.
        """
        image_embeds_proj = self.projector(image_embeds)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_embeds_proj, text_embeds], dim=1)
        
        # generate() 함수는 inputs_embeds 입력을 지원합니다.
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True, # 창의적인 생성을 위해 샘플링 사용
            temperature=0.7
        )
        
        # 생성된 토큰을 텍스트로 디코딩
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)