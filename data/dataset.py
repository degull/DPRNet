# ==============================================================================
# 📋 DPR-Net V2 Dataset Loader
# 역할: 이미지와 캡션을 로드하고, LLM과 Vision Encoder에 맞게 전처리합니다.
# 핵심: 가변 길이 텍스트 처리 및 Attention Mask 2D 정렬 (Vision + Text)
# ==============================================================================

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
import torchvision.transforms as transforms

class DPRDataset(Dataset):
    def __init__(self, data_root, metadata_file, tokenizer_path="mistralai/Mistral-7B-v0.1", img_size=224, max_length=128):
        self.data_root = data_root
        self.img_size = img_size
        self.max_length = max_length
        
        # Metadata 로드
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.image_paths = list(self.metadata.keys())
        
        # Tokenizer & Processor
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # VETNet용 전처리 (학습을 위해 256x256 Resize로 통일)
        self.transform_vet = transforms.Compose([
            #transforms.Resize(256),
            transforms.RandomCrop(128),
            transforms.ToTensor() 
        ])

    def _get_gt_path(self, input_path):
        """
        입력 이미지 경로를 바탕으로 정답(Ground Truth) 이미지 경로를 추론합니다.
        """
        gt_path = input_path
        
        # 1. Rain100H (rain -> norain)
        if 'rain' in input_path and 'norain' not in input_path:
            gt_path = input_path.replace('rain', 'norain')
        
        # 2. LOL Dataset (low -> high)
        elif 'low' in input_path:
            gt_path = input_path.replace('low', 'high')
            
        # 3. CSD (Snow -> Gt)
        elif 'Snow' in input_path:
            gt_path = input_path.replace('Snow', 'Gt')
            
        # 4. SOTS (hazy -> clear)
        elif 'hazy' in input_path:
            gt_path = input_path.replace('hazy', 'clear')
            
        # 파일 확인
        if not os.path.exists(gt_path):
            # GT가 없으면 자기 자신을 반환 (Self-Supervised 대비 혹은 에러 방지)
            # print(f"⚠️ Warning: GT not found for {input_path}")
            return input_path
            
        return gt_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.metadata[img_path]
        
        # GT 경로 찾기
        gt_path = self._get_gt_path(img_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            gt_image = Image.open(gt_path).convert('RGB')
        except:
            # 로드 실패 시 다음 이미지로 넘어감
            return self.__getitem__((idx + 1) % len(self))

        # 1. CLIP Input (Vision Encoder용)
        pixel_values = self.clip_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # 2. VETNet Input & Target (복원 네트워크용)
        vet_input = self.transform_vet(image)   # 손상된 이미지 (Input)
        vet_target = self.transform_vet(gt_image) # 정답 이미지 (Target)
        
        # 3. Text Input (Mistral용)
        text_inputs = self.tokenizer(
            caption, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length, add_special_tokens=True
        )
        input_ids = text_inputs.input_ids.squeeze(0)
        text_mask = text_inputs.attention_mask.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "text_mask": text_mask,
            "vet_input": vet_input,
            "vet_target": vet_target
        }

    @staticmethod
    def collate_fn(batch):
        # 배치 데이터 스택
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        text_mask = torch.stack([item['text_mask'] for item in batch])
        
        # VETNet 데이터 스택
        vet_input = torch.stack([item['vet_input'] for item in batch])
        vet_target = torch.stack([item['vet_target'] for item in batch])
        
        batch_size = len(batch)
        
        # Vision Mask 생성 (모두 1)
        vision_mask = torch.ones((batch_size, 257), dtype=torch.long)
        
        # 전체 Attention Mask 결합 [Vision | Text]
        attention_mask = torch.cat([vision_mask, text_mask], dim=1)
        
        return {
            "pixel_values": pixel_values,      # [B, 3, 224, 224]
            "input_ids": input_ids,            # [B, Text_Len]
            "attention_mask": attention_mask,  # [B, 257+Text_Len]
            "vet_input": vet_input,            # [B, 3, 256, 256] -> 모델 입력
            "vet_target": vet_target           # [B, 3, 256, 256] -> Loss 계산용 정답
        }