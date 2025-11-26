import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
import torchvision.transforms as transforms

# ==============================================================================
# 📋 DPR-Net V2 Dataset Loader
# 역할: 이미지와 캡션을 로드하고, LLM과 Vision Encoder에 맞게 전처리합니다.
# 핵심: 가변 길이 텍스트 처리 및 Attention Mask 2D 정렬 (Vision + Text)
# ==============================================================================

class DPRDataset(Dataset):
    def __init__(self, data_root, metadata_file, tokenizer_path="mistralai/Mistral-7B-v0.1", img_size=224, max_length=128):
        """
        Args:
            data_root (str): 데이터셋 루트 폴더
            metadata_file (str): 전처리된 JSON 파일 경로 (preprocess_captions.py 결과물)
            tokenizer_path (str): Mistral Tokenizer 경로
            img_size (int): CLIP/LLM 입력용 이미지 크기 (기본 224)
            max_length (int): 텍스트 최대 토큰 길이
        """
        self.data_root = data_root
        self.img_size = img_size
        self.max_length = max_length
        
        print(f"📂 Loading Metadata from: {metadata_file}")
        # Metadata 로드
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 유효한 이미지 경로만 필터링 (파일이 실제로 존재하는지 확인)
        self.image_paths = []
        valid_count = 0
        for path in self.metadata.keys():
            if os.path.exists(path):
                self.image_paths.append(path)
                valid_count += 1
            # 윈도우 경로 호환성 체크 (혹시 경로가 다를 경우 대비)
            elif os.path.exists(path.replace('\\', '/')):
                 self.image_paths.append(path.replace('\\', '/'))
                 valid_count += 1
        
        print(f"   ✅ Loaded {valid_count} valid image-caption pairs.")
        
        # Tokenizer 설정
        print(f"🤖 Loading Tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Mistral은 기본 pad_token이 없으므로 eos_token을 pad로 설정
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        # 1. Vision Encoder용 전처리 (CLIP 통계량 정규화)
        # CLIP은 내부적으로 RGB 입력을 기대하며, 특정 mean/std를 사용함
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # 2. VETNet용 고해상도 이미지 전처리 (Tensor 변환만 수행)
        self.transform_highres = transforms.Compose([
            transforms.ToTensor() 
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.metadata[img_path] # JSON에서 캡션 가져오기
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading image {img_path}: {e}")
            # 에러 발생 시 다음 이미지를 가져오도록 재귀 호출
            return self.__getitem__((idx + 1) % len(self))

        # [A] Vision Inputs Preparation (For CLIP)
        # CLIPProcessor가 Resize 및 Normalize를 자동 처리해줌
        # return_tensors='pt' -> [1, 3, 224, 224] -> squeeze -> [3, 224, 224]
        pixel_values = self.clip_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # [B] High-Res Image (For VETNet Input/Target)
        # 원본 해상도를 유지하거나 학습 시 RandomCrop 등을 적용할 수 있음
        # 여기서는 전체 이미지를 텐서로 변환
        high_res_image = self.transform_highres(image)
        
        # [C] Text Inputs Preparation (For Mistral)
        # ⚠️ 가변 길이 처리: 
        # 여기서는 max_length로 자르되, 패딩은 배치 단위로 collate_fn에서 처리하거나
        # 여기서 'max_length'로 통일할 수 있음. (메모리 효율을 위해 여기서는 max_length 고정 사용)
        text_inputs = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length", # 모든 샘플 길이를 128로 고정 (배치 처리가 쉬움)
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        input_ids = text_inputs.input_ids.squeeze(0)      # [max_len]
        text_mask = text_inputs.attention_mask.squeeze(0) # [max_len] (1: text, 0: pad)

        return {
            "img_path": img_path,
            "pixel_values": pixel_values,      # [3, 224, 224] for CLIP
            "high_res_image": high_res_image,  # [3, H, W] for VETNet
            "input_ids": input_ids,            # [max_len] for Mistral
            "text_mask": text_mask,            # [max_len] for Mistral Masking
            "raw_caption": caption             # 디버깅 및 시각화용
        }

    # ==========================================================================
    # ⚠️ Custom Collate Function (배치 처리의 핵심)
    # 역할: 여러 샘플을 묶어 하나의 배치 텐서로 만들고, 2D Attention Mask를 생성
    # ==========================================================================
    @staticmethod
    def collate_fn(batch):
        # 1. 배치 내 데이터 스택 (Stacking)
        pixel_values = torch.stack([item['pixel_values'] for item in batch]) # [B, 3, 224, 224]
        input_ids = torch.stack([item['input_ids'] for item in batch])       # [B, max_len]
        text_mask = torch.stack([item['text_mask'] for item in batch])       # [B, max_len]
        
        # High-Res 이미지는 크기가 다를 수 있으므로 리스트로 유지하거나, 
        # 학습 단계에서 RandomCrop으로 크기를 맞춰야 stack 가능. 
        # (일단 리스트로 반환하여 모델 내부에서 처리하도록 함)
        high_res_images = [item['high_res_image'] for item in batch]
        raw_captions = [item['raw_caption'] for item in batch]
        img_paths = [item['img_path'] for item in batch]
        
        batch_size = len(batch)
        vision_token_len = 257 # CLIP ViT-L/14: 1 (CLS) + 256 (Patches)
        
        # -----------------------------------------------------------
        # ⭐ 명세서 핵심: Attention Mask 생성 [Batch, 257 + Text_Len]
        # Vision 파트는 무조건 1 (모두 보임), Text 파트는 패딩에 따라 0 또는 1
        # -----------------------------------------------------------
        
        # (A) Vision Mask 생성: [Batch, 257] -> 모두 1 (Attention 가능)
        vision_mask = torch.ones((batch_size, vision_token_len), dtype=torch.long)
        
        # (B) Full Mask 결합: [Vision(1) | Text(0 or 1)]
        # text_mask는 이미 [B, Text_Len]이며 패딩이 0으로 되어있음.
        attention_mask = torch.cat([vision_mask, text_mask], dim=1)
        
        return {
            "pixel_values": pixel_values,      # [B, 3, 224, 224] -> CLIP 입력
            "input_ids": input_ids,            # [B, Text_Len]    -> Mistral 입력
            "attention_mask": attention_mask,  # [B, 257 + Text_Len] -> Mistral Mask
            "high_res_images": high_res_images,# List[Tensor]     -> VETNet 입력 (or Ground Truth)
            "raw_captions": raw_captions,      # List[str]        -> 시각화용
            "img_paths": img_paths             # List[str]        -> 경로 추적
        }