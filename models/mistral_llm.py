import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# ==============================================================================
# 🧠 The Brain: Mistral-7B LLM (4-bit Quantized + FP32 Adapters)
# ==============================================================================

class MistralLLM(nn.Module):
    def __init__(self, model_id="mistralai/Mistral-7B-v0.1", vision_hidden_size=1024, llm_hidden_size=4096):
        super().__init__()
        print(f"🧠 Loading Mistral LLM (4-bit): {model_id}...")
        
        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        # 4-bit Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # 2. Load Model in 4-bit
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # 학습 안정화를 위한 전처리 (LayerNorm 등을 FP32로 변환)
        self.llm = prepare_model_for_kbit_training(self.llm)

        # 3. Projector (Adapter) - ⚠️ 수정됨!
        # GradScaler 호환성을 위해 학습 가능한 레이어는 반드시 Float32여야 함
        self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)
        self.projector.float() # <--- .half() 대신 .float() 사용!
        
        # 4. Apply LoRA
        print("🚀 Applying LoRA (Low-Rank Adaptation)...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        
        # ✅ 안전장치: 모든 학습 가능한 파라미터를 FP32로 강제 변환 (GradScaler 에러 방지)
        for param in self.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def forward(self, image_embeds, input_ids, attention_mask):
        # 1. Vision Projection
        # 입력이 FP16이어도 레이어가 FP32면 알아서 오토캐스트됨
        image_embeds_proj = self.projector(image_embeds)
        
        # 2. Text Embedding
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 3. Concatenation
        inputs_embeds = torch.cat([image_embeds_proj, text_embeds], dim=1)
        
        if inputs_embeds.requires_grad:
            inputs_embeds.retain_grad()
        
        # 4. Forward Pass
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 5. Return Last Hidden State
        return outputs.hidden_states[-1]

    def generate_caption(self, image_embeds, input_ids, max_new_tokens=50):
        image_embeds_proj = self.projector(image_embeds)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_embeds_proj, text_embeds], dim=1)
        
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True, 
            temperature=0.7
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)