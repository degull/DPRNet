import os
import yaml
import torch
import numpy as np
from PIL import Image

# --- 사용자 모듈 임포트 ---
try:
    from model.dpr_net import DPR_Net
except ImportError as e:
    print(f"Error: 모델 패키지 로드 실패 ({e}).")
    exit(1)

def check_mistral_thought_fix(config_path, checkpoint_path, image_path):
    print(f"\n--- 🕵️‍♂️ Checking Mistral's Thoughts (Fixed Mode) ---")
    print(f"Target Image: {image_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 설정 및 모델 로드
    if not os.path.exists(config_path): return
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    cfg_model_vet = config['model']['vetnet']
    cfg_model_clip = config['model']['clip']
    cfg_model_llm = config['model']['llm']

    print("Initializing Model...")
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

    # 2. 체크포인트 로드
    if os.path.exists(checkpoint_path):
        print(f"Loading Checkpoint: {os.path.basename(checkpoint_path)}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("Error: Checkpoint not found!")
        return

    model.eval()

    # 3. 이미지 전처리
    try:
        raw_image = Image.open(image_path).convert('RGB').resize((224, 224))
        input_tensor = torch.from_numpy(np.array(raw_image)).float() / 255.0
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        input_tensor = (input_tensor - mean) / std
    except Exception as e:
        print(f"Image Error: {e}")
        return

    # 4. 텍스트 생성 (Generate with strict limit)
    print("Thinking... (Generating Text with Limit)")
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # A. Vision Features Extraction
            vision_tokens, _ = model.clip_encoder(input_tensor)
            vision_embeds = model.llm_module.input_projector(vision_tokens)
            
            # B. Attention Mask 생성 (필수!)
            batch_size, seq_len, _ = vision_embeds.shape
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

            # C. Generate (강제 종료 조건 추가)
            generated_ids = model.llm_module.llm.generate(
                inputs_embeds=vision_embeds,
                attention_mask=attention_mask,
                max_new_tokens=64,       # 최대 64 토큰까지만 생성 (약 30~40단어)
                min_new_tokens=5,        # 최소 5 토큰
                do_sample=False,         # Greedy Search (가장 확률 높은 단어 선택 -> 빠름)
                num_beams=1,             # Beam Search 끔 (메모리 절약, 속도 향상)
                pad_token_id=model.tokenizer.eos_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                repetition_penalty=1.2   # 같은 말 반복 방지
            )

    # 5. 디코딩 및 출력
    print("Decoding...")
    decoded_text = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n" + "="*60)
    print(f"🖼️  Analysis of: {os.path.basename(image_path)}")
    print("-" * 60)
    print(f"🤖 [Mistral's Thought]:\n")
    print(f"\"{decoded_text.strip()}\"")
    print("="*60 + "\n")

if __name__ == "__main__":
    CONFIG_PATH = "configs/dpr_config.yaml"
    CHECKPOINT_PATH = r"G:\DPR-Net\checkpoints\epoch_5_total_0_0722_img_0_0620_text_0_0096_cons_0_0558.pth"
    TEST_IMAGE = r"G:\DPR-Net\data\rain100H\test\rain\norain-1.png"
    
    if os.path.exists(TEST_IMAGE):
        check_mistral_thought_fix(CONFIG_PATH, CHECKPOINT_PATH, TEST_IMAGE)
    else:
        print(f"Image not found: {TEST_IMAGE}")