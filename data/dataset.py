import os
import glob
import random
import io 

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image 
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer

# -----------------------------------------------
#  (Pipeline B) 하이브리드 왜곡 함수
# -----------------------------------------------
def add_gaussian_noise(img_tensor, mean=0., std_range=(0.01, 0.2)):
    """ 텐서에 랜덤 가우시안 노이즈 추가 """
    std = random.uniform(std_range[0], std_range[1])
    noise = torch.randn_like(img_tensor) * std + mean
    return torch.clamp(img_tensor + noise, 0.0, 1.0)
def add_gaussian_blur(img_tensor, kernel_range=(3, 9), sigma_range=(0.1, 2.0)):
    """ 텐서에 랜덤 가우시안 블러 추가 """
    kernel_size = random.choice(list(range(kernel_range[0], kernel_range[1] + 1, 2)))
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    return T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img_tensor)
def add_jpeg_compression(img_tensor, quality_range=(30, 90)):
    """ 텐서에 랜덤 JPEG 압축 손상 추가 """
    quality = random.randint(quality_range[0], quality_range[1])
    pil_img = T.ToPILImage()(img_tensor)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    pil_img_compressed = Image.open(buffer)
    return T.ToTensor()(pil_img_compressed)


# -----------------------------------------------
#  메인 DPR-Net 데이터셋 클래스 (LoRA + 영어 프롬프트)
# -----------------------------------------------
class DPRDataset(Dataset):
    """
    [수정] (img, clean, text_labels, text_mask) 4개 항목을 반환
    """
    
    def __init__(self, data_dir, tokenizer, mode='train', 
                 patch_size=128, hybrid_prob=0.5, 
                 max_text_len=128): # (영어도 128로 충분)
        super().__init__()
        
        self.data_dir = data_dir
        self.mode = mode
        self.patch_size = patch_size
        self.hybrid_prob = hybrid_prob if mode == 'train' else 0.0 
        
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        
        # --- [수정] (중요) LLM에게 가르칠 "영어" 정답 텍스트 ---
        self.prompt_templates = {
            "CSD": (
                "Diagnosis: High-density snow particles detected across the image.\n"
                "Action: Activating desnowing control signals.\n"
                "(Deeper Stages) Restoring global structure and contrast via Global Token FiLM.\n"
                "(Shallow Stages) Detecting snow patterns via Patch Tokens for inpainting."
            ),
            "LOL": (
                "Diagnosis: Severe low-light conditions and heavy sensor noise detected.\n"
                "Action: Activating denoising and contrast enhancement signals.\n"
                "(Deeper Stages) Restoring global exposure and contrast via Global Token FiLM.\n"
                "(Shallow Stages) Detecting noise patterns via Patch Tokens for pixel smoothing."
            ),
            "Rain100H": (
                "Diagnosis: Heavy rain streaks identified across the image.\n"
                "Action: Activating deraining control signals.\n"
                "(Deeper Stages) Restoring global saturation degraded by rain via Global Token FiLM.\n"
                "(Shallow Stages) Detecting streak patterns via Patch Tokens to guide inpainting."
            ),
            "SOTS-indoor": (
                "Diagnosis: Indoor image with significant low-contrast haze detected.\n"
                "Action: Activating dehazing control signals.\n"
                "(Deeper Stages) Transmitting Global Token based FiLM signals to restore overall image contrast and saturation."
            ),
            "SOTS-outdoor": (
                "Diagnosis: Outdoor image with significant low-contrast haze detected.\n"
                "Action: Activating dehazing control signals.\n"
                "(Deeper Stages) Transmitting Global Token based FiLM signals to restore overall image contrast and saturation."
            ),
            "Hybrid": (
                "Diagnosis: Complex hybrid degradation detected.\n"
                "(1) Global low-contrast Haze and\n"
                "(2) Light Rain streaks are identified.\n"
                "Action: Generating composite control signals for simultaneous dehazing and deraining.\n"
                "(Deeper Stages) Transmitting Global Token FiLM for haze removal (contrast/saturation).\n"
                "(Shallow Stages) Transmitting Patch Token FiLM for rain streak removal (inpainting)."
            )
        }
        
        self.image_pairs = self._load_all_image_pairs()
        
        if len(self.image_pairs) == 0:
            raise FileNotFoundError(f"'{mode}' 모드에 대한 이미지 쌍을 찾을 수 없습니다.")
            
        print(f"Loaded {len(self.image_pairs)} image pairs for mode '{mode}'.")
        if self.mode == 'train':
            print(f"  (Hybrid Synthesis Probability: {self.hybrid_prob*100}%)")
            print(f"  (Max Text Length: {self.max_text_len})")

    def _load_all_image_pairs(self):
        # (이 함수는 변경 사항 없음)
        pairs = []
        # --- 1. CSD ---
        csd_mode_path = 'Train' if self.mode == 'train' else 'Test'
        csd_distorted_path = os.path.join(self.data_dir, 'CSD', csd_mode_path, 'Snow')
        csd_clean_path = os.path.join(self.data_dir, 'CSD', csd_mode_path, 'Gt')
        csd_distorted_files = sorted(glob.glob(os.path.join(csd_distorted_path, "*.tif")))
        for dist_file in csd_distorted_files:
            filename = os.path.basename(dist_file)
            clean_file = os.path.join(csd_clean_path, filename)
            if os.path.exists(clean_file):
                pairs.append((dist_file, clean_file, "CSD")) 
        # --- 2. LOL ---
        lol_mode_path = 'our485' if self.mode == 'train' else 'eval15'
        lol_distorted_path = os.path.join(self.data_dir, 'lol_dataset', lol_mode_path, 'low')
        lol_clean_path = os.path.join(self.data_dir, 'lol_dataset', lol_mode_path, 'high')
        lol_distorted_files = sorted(glob.glob(os.path.join(lol_distorted_path, "*.*")))
        for dist_file in lol_distorted_files:
            filename = os.path.basename(dist_file)
            clean_file = os.path.join(lol_clean_path, filename)
            if os.path.exists(clean_file):
                pairs.append((dist_file, clean_file, "LOL")) 
        # --- 3. Rain100H ---
        rain_mode_path = 'train' if self.mode == 'train' else 'test'
        rain_distorted_path = os.path.join(self.data_dir, 'rain100H', rain_mode_path, 'rain')
        rain_clean_path = os.path.join(self.data_dir, 'rain100H', rain_mode_path, 'norain')
        rain_distorted_files = sorted(glob.glob(os.path.join(rain_distorted_path, "*.*")))
        for dist_file in rain_distorted_files:
            filename = os.path.basename(dist_file)
            clean_file = os.path.join(rain_clean_path, filename)
            if os.path.exists(clean_file):
                pairs.append((dist_file, clean_file, "Rain100H"))
        # --- 4. SOTS ---
        sots_types = ['indoor', 'outdoor']
        for sots_type in sots_types:
            sots_distorted_path = os.path.join(self.data_dir, 'SOTS', sots_type, 'hazy')
            sots_clean_path = os.path.join(self.data_dir, 'SOTS', sots_type, 'clear')
            sots_distorted_files = sorted(glob.glob(os.path.join(sots_distorted_path, "*.*")))
            for dist_file in sots_distorted_files:
                filename = os.path.basename(dist_file)
                base_filename_no_ext = filename.split('_')[0]
                base_filename = base_filename_no_ext + ".png" 
                clean_file = os.path.join(sots_clean_path, base_filename)
                if os.path.exists(clean_file):
                    pairs.append((dist_file, clean_file, f"SOTS-{sots_type}")) 
        return pairs

    def __len__(self):
        # (오류 수정을 위해 __len__ 함수 추가)
        return len(self.image_pairs)

    def _get_transforms(self, distorted_img_pil, clean_img_pil):
        # (이 함수는 변경 사항 없음)
        distorted_tensor = TF.to_tensor(distorted_img_pil)
        clean_tensor = TF.to_tensor(clean_img_pil)
        _, h, w = distorted_tensor.shape
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        if pad_h > 0 or pad_w > 0:
            padding = (0, 0, pad_w, pad_h) 
            distorted_tensor = TF.pad(distorted_tensor, padding=padding, fill=0)
            clean_tensor = TF.pad(clean_tensor, padding=padding, fill=0)
        i, j, h, w = T.RandomCrop.get_params(
            distorted_tensor, output_size=(self.patch_size, self.patch_size)
        )
        distorted_tensor = TF.crop(distorted_tensor, i, j, h, w)
        clean_tensor = TF.crop(clean_tensor, i, j, h, w)
        if random.random() > 0.5:
            distorted_tensor = TF.hflip(distorted_tensor)
            clean_tensor = TF.hflip(clean_tensor)
        return distorted_tensor, clean_tensor

    def __getitem__(self, idx):
        distorted_path, clean_path, dataset_name = self.image_pairs[idx]
        
        try:
            distorted_img_pil = Image.open(distorted_path).convert('RGB')
            clean_img_pil = Image.open(clean_path).convert('RGB')
            dist_tensor_orig, clean_tensor = self._get_transforms(distorted_img_pil, clean_img_pil)

            prompt_text = ""
            
            if random.random() < self.hybrid_prob:
                hybrid_augs = [add_gaussian_blur, add_gaussian_noise, add_jpeg_compression]
                random.shuffle(hybrid_augs)
                num_augs = random.randint(1, 3) 
                dist_tensor_final = clean_tensor
                for aug in hybrid_augs[:num_augs]:
                    dist_tensor_final = aug(dist_tensor_final)
                prompt_text = self.prompt_templates["Hybrid"]
            else:
                dist_tensor_final = dist_tensor_orig
                prompt_text = self.prompt_templates.get(dataset_name, "Diagnosis: General degradation detected. Action: Applying standard restoration.")

            # --- [수정] 정답 텍스트 토큰화 ---
            prompt_with_eos = prompt_text + self.tokenizer.eos_token
            
            tokenizer_output = self.tokenizer(
                prompt_with_eos,
                padding='max_length',     
                truncation=True,          
                max_length=self.max_text_len, 
                return_tensors="pt"
            )
            
            text_labels = tokenizer_output.input_ids.squeeze(0)
            text_mask = tokenizer_output.attention_mask.squeeze(0)
            
            return dist_tensor_final, clean_tensor, text_labels, text_mask

        except Exception as e:
            print(f"Error loading image or tokenizing: {distorted_path} (Dataset: {dataset_name})")
            print(f"Error details: {e}")
            dummy_text = torch.zeros(self.max_text_len, dtype=torch.long)
            dummy_mask = torch.zeros(self.max_text_len, dtype=torch.long)
            return torch.zeros(3, self.patch_size, self.patch_size), \
                   torch.zeros(3, self.patch_size, self.patch_size), \
                   dummy_text, dummy_mask

# -----------------------------------------------
#  테스트 코드
# -----------------------------------------------
if __name__ == "__main__":
    
    print("--- [Test] DPRDataset (LoRA Version) ---")
    print("[Note] 이 파일은 단독으로 실행할 수 없습니다.")
    print("train.py가 LLM 토크나이저를 로드한 후 이 클래스를 초기화해야 합니다.")
    
    try:
        print("Loading temporary tokenizer for test...")
        tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            
        DATA_DIR = r"G:\DPR-Net\data" 
        PATCH_SIZE = 128
        MAX_TEXT_LEN = 128 # (수정한 길이로 테스트)
        
        print("\n--- Testing 'train' mode (LoRA) ---")
        train_dataset = DPRDataset(
            data_dir=DATA_DIR, 
            tokenizer=tok, 
            mode='train', 
            patch_size=PATCH_SIZE, 
            hybrid_prob=0.5,
            max_text_len=MAX_TEXT_LEN
        )
        
        if len(train_dataset) > 0:
            print(f"Found {len(train_dataset)} training samples.")
            dist, clean, labels, mask = train_dataset[0]
            print(f"  Sample 0: Dist={dist.shape}, Clean={clean.shape}")
            print(f"  Sample 0: Labels={labels.shape}, Mask={mask.shape}")
            print(f"  Decoded Labels (first 10 tokens): {tok.decode(labels[:10], skip_special_tokens=False)}")
            
            assert dist.shape == (3, PATCH_SIZE, PATCH_SIZE)
            assert labels.shape == (MAX_TEXT_LEN,) 
            
            print("\n✅ 'train' mode (LoRA) test passed.")
        else:
            print("⚠️  'train' dataset is EMPTY.")
            
    except Exception as e:
        import traceback
        print(f"\n[Fatal Error] An unexpected error occurred:")
        traceback.print_exc()