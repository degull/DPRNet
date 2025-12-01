# G:\DPR-Net\models\mistral_llm.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# ==============================================================================
# 🧠 The Brain: Mistral-7B LLM (4-bit Quantized + FP32 Adapters)
# ==============================================================================

class MistralLLM(nn.Module):
    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-v0.1",
        vision_hidden_size: int = 1024,
        llm_hidden_size: int = 4096,
    ):
        super().__init__()
        print(f"🧠 Loading Mistral LLM (4-bit): {model_id}...")

        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            # Use EOS as PAD (common for causal LMs)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 4-bit Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # 3. Load Model in 4-bit
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # 학습 안정화를 위한 전처리 (LayerNorm 등을 FP32로 변환)
        self.llm = prepare_model_for_kbit_training(self.llm)

        # 4. Projector (Adapter): CLIP(1024) -> Mistral(4096)
        # GradScaler 호환성을 위해 학습 가능한 레이어는 FP32로 둔다.
        self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)
        self.projector.float()

        # 5. Apply LoRA
        print("🚀 Applying LoRA (Low-Rank Adaptation)...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()

        # ✅ 안전장치: 모든 학습 가능한 파라미터를 FP32로 강제 변환 (GradScaler 에러 방지)
        for param in self.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def forward(self, image_embeds, input_ids, attention_mask):
        """
        image_embeds: (B, 257, vision_hidden_size=1024) from CLIP
        input_ids:    (B, T) text token ids
        attention_mask: (B, 257 + T) where:
            - vision tokens (0..256) = 1
            - text tokens   (257..257+L-1) = 1
            - pad region    (else) = 0
        Returns:
            last_hidden_state: (B, 257 + T, 4096)
        """
        # 1) Vision Projection: 1024 -> 4096
        image_embeds_proj = self.projector(image_embeds)  # (B, 257, 4096)

        # 2) Text Embedding
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, T, 4096)

        # 3) Concatenate along sequence dimension
        inputs_embeds = torch.cat([image_embeds_proj, text_embeds], dim=1)  # (B, 257+T, 4096)

        # 🔥 Gradient chain 유지: PixelLM → Pixel Decoder → FiLM → VETNet까지 end-to-end
        inputs_embeds.requires_grad_(True)

        if inputs_embeds.requires_grad:
            # 디버깅용: LLM 입력 임베딩의 gradient를 확인할 수 있도록 retain
            inputs_embeds.retain_grad()

        # 4) Position IDs (0,1,2,..., 257+T-1)
        #    Vision 토큰 257개는 항상 0~256, 텍스트는 그 이후로 정렬
        seq_len = inputs_embeds.size(1)
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)  # (1, 257+T)

        # 5) Sanity check: attention_mask 길이가 inputs_embeds와 정확히 일치하는지
        assert (
            attention_mask.shape[1] == seq_len
        ), f"[MistralLLM] attention_mask length mismatch: {attention_mask.shape} vs {inputs_embeds.shape}"

        # 6) Mistral Forward (LoRA 학습)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        # 7) Return last hidden state (B, 257+T, 4096)
        return outputs.hidden_states[-1]

    def generate_caption(self, image_embeds, input_ids, max_new_tokens: int = 50):
        """
        설명/디버그 용 캡션 생성 모드.
        이미지 + 텍스트 프롬프트를 기반으로 Mistral generate() 호출.
        """
        # 1) Vision Projection
        image_embeds_proj = self.projector(image_embeds)  # (B, 257, 4096)

        # 2) Text Embedding
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, T, 4096)

        # 3) Concatenate
        inputs_embeds = torch.cat([image_embeds_proj, text_embeds], dim=1)  # (B, 257+T, 4096)

        batch_size, seq_len, _ = inputs_embeds.shape
        vision_len = image_embeds_proj.size(1)
        text_len = text_embeds.size(1)

        # 4) Attention mask: vision=1, text(non-pad)=1, text-pad=0
        text_mask = (input_ids != self.tokenizer.pad_token_id).long()  # (B, T)
        vision_mask = torch.ones(batch_size, vision_len, device=inputs_embeds.device, dtype=torch.long)
        attention_mask = torch.cat([vision_mask, text_mask], dim=1)  # (B, 257+T)

        # 5) Position IDs
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)  # (1, 257+T)

        # 6) Generate text (설명 모드)
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            position_ids=position_ids,  # model_kwargs로 forward에 전달됨
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
