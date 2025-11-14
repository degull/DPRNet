# G:\DPR-Net\data\dataset.py
import os
import glob
import random
import io  # JPEG 인메모리 압축에 필요

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image  # .tif 및 .png, .jpg 모두 처리가능
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# -----------------------------------------------
#  (Pipeline B) 하이브리드 왜곡 함수
# -----------------------------------------------
# (모든 함수는 0~1 범위의 PyTorch Tensor를 입력받습니다)

def add_gaussian_noise(img_tensor, mean=0., std_range=(0.01, 0.2)):
    """ 텐서에 랜덤 가우시안 노이즈 추가 """
    std = random.uniform(std_range[0], std_range[1])
    noise = torch.randn_like(img_tensor) * std + mean
    return torch.clamp(img_tensor + noise, 0.0, 1.0)

def add_gaussian_blur(img_tensor, kernel_range=(3, 9), sigma_range=(0.1, 2.0)):
    """ 텐서에 랜덤 가우시안 블러 추가 """
    # 커널 크기는 홀수여야 함
    kernel_size = random.choice(list(range(kernel_range[0], kernel_range[1] + 1, 2)))
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    return T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img_tensor)

def add_jpeg_compression(img_tensor, quality_range=(30, 90)):
    """ 텐서에 랜덤 JPEG 압축 손상 추가 """
    quality = random.randint(quality_range[0], quality_range[1])
    
    # Tensor -> PIL Image
    pil_img = T.ToPILImage()(img_tensor)
    
    # PIL -> In-memory JPEG Buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    
    # Buffer -> PIL Image
    pil_img_compressed = Image.open(buffer)
    
    # PIL -> Tensor
    return T.ToTensor()(pil_img_compressed)


# -----------------------------------------------
#  메인 DPR-Net 데이터셋 클래스
# -----------------------------------------------

class DPRDataset(Dataset):
    """
    DPR-Net "Dual-Pipeline" Dataset.
    
    - Pipeline A (Standard Pair): 디스크의 (왜곡, 정답) 쌍 로드
    - Pipeline B (Hybrid Pair): 정답 이미지에 실시간 하이브리드 왜곡 합성
    """
    
    def __init__(self, data_dir, mode='train', patch_size=128, hybrid_prob=0.5):
        """
        Args:
            data_dir (str): 'G:\DPR-Net\data' 경로
            mode (str): 'train' 또는 'test' (test 모드는 hybrid_prob=0)
            patch_size (int): 훈련용 랜덤 크롭 크기
            hybrid_prob (float): Pipeline B (하이브리드)를 사용할 확률
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.mode = mode
        self.patch_size = patch_size
        
        # 'test' 모드에서는 하이브리드 합성을 비활성화 (순수 성능 측정)
        self.hybrid_prob = hybrid_prob if mode == 'train' else 0.0 
        
        # self.image_pairs = [(dist_path, clean_path, "dataset_name"), ...]
        self.image_pairs = self._load_all_image_pairs()
        
        if len(self.image_pairs) == 0:
            raise FileNotFoundError(f"'{mode}' 모드에 대한 이미지 쌍을 찾을 수 없습니다. {data_dir} 경로와 폴더 구조를 확인하세요.")
            
        print(f"Loaded {len(self.image_pairs)} image pairs for mode '{mode}'.")
        if self.mode == 'train':
            print(f"  (Hybrid Synthesis Probability: {self.hybrid_prob*100}%)")

    def _load_all_image_pairs(self):
        """ 제공된 모든 데이터셋 폴더를 스캔하여 (왜곡, 정답) 쌍 목록 생성 """
        pairs = []
        
        # --- 1. CSD (Snow, .tif) ---
        # G:\DPR-Net\data\CSD
        csd_mode_path = 'Train' if self.mode == 'train' else 'Test'
        csd_distorted_path = os.path.join(self.data_dir, 'CSD', csd_mode_path, 'Snow')
        csd_clean_path = os.path.join(self.data_dir, 'CSD', csd_mode_path, 'Gt')
        
        # .tif 파일 검색
        csd_distorted_files = sorted(glob.glob(os.path.join(csd_distorted_path, "*.tif")))
        for dist_file in csd_distorted_files:
            filename = os.path.basename(dist_file)
            clean_file = os.path.join(csd_clean_path, filename)
            if os.path.exists(clean_file):
                pairs.append((dist_file, clean_file, "CSD"))
            else:
                print(f"[CSD Warning] Missing clean file for: {filename}")


        # --- 2. LOL (Low-Light) ---
        # G:\DPR-Net\data\lol_dataset
        lol_mode_path = 'our485' if self.mode == 'train' else 'eval15'
        lol_distorted_path = os.path.join(self.data_dir, 'lol_dataset', lol_mode_path, 'low')
        lol_clean_path = os.path.join(self.data_dir, 'lol_dataset', lol_mode_path, 'high')

        # 모든 이미지 확장자(*.*) 검색
        lol_distorted_files = sorted(glob.glob(os.path.join(lol_distorted_path, "*.*")))
        for dist_file in lol_distorted_files:
            filename = os.path.basename(dist_file)
            clean_file = os.path.join(lol_clean_path, filename)
            if os.path.exists(clean_file):
                pairs.append((dist_file, clean_file, "LOL"))

        # --- 3. Rain100H (Rain) ---
        # G:\DPR-Net\data\rain100H
        rain_mode_path = 'train' if self.mode == 'train' else 'test'
        rain_distorted_path = os.path.join(self.data_dir, 'rain100H', rain_mode_path, 'rain')
        rain_clean_path = os.path.join(self.data_dir, 'rain100H', rain_mode_path, 'norain')

        rain_distorted_files = sorted(glob.glob(os.path.join(rain_distorted_path, "*.*")))
        for dist_file in rain_distorted_files:
            filename = os.path.basename(dist_file)
            # Rain100H는 'rain' 폴더와 'norain' 폴더의 파일명이 동일함
            clean_file = os.path.join(rain_clean_path, filename)
            if os.path.exists(clean_file):
                pairs.append((dist_file, clean_file, "Rain100H"))
        
        # --- 4. SOTS (Haze) ---
        # G:\DPR-Net\data\SOTS
        # (SOTS는 보통 훈련/테스트가 indoor/outdoor로 나뉘므로, 둘 다 사용)
        sots_types = ['indoor', 'outdoor']
        for sots_type in sots_types:
            sots_distorted_path = os.path.join(self.data_dir, 'SOTS', sots_type, 'hazy')
            sots_clean_path = os.path.join(self.data_dir, 'SOTS', sots_type, 'clear')
            
            sots_distorted_files = sorted(glob.glob(os.path.join(sots_distorted_path, "*.*")))
            for dist_file in sots_distorted_files:
                filename = os.path.basename(dist_file)
                
                # SOTS 이름 규칙:
                # (outdoor) 0001_0.8_0.2.jpg -> 0001
                # (indoor)  1400_1.png       -> 1400
                base_filename_no_ext = filename.split('_')[0]
                
                # SOTS 'clear' 파일은 확장자가 .png (indoor) 또는 .png (outdoor) 임
                base_filename = base_filename_no_ext + ".png" 
                
                clean_file = os.path.join(sots_clean_path, base_filename)
                
                if os.path.exists(clean_file):
                    pairs.append((dist_file, clean_file, f"SOTS-{sots_type}"))
                else:
                    # (실제로 파일이 없는 경우에만 경고)
                    print(f"[SOTS Warning] Missing clean file: {clean_file} (for {dist_file})")
                        
        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def _get_transforms(self, distorted_img_pil, clean_img_pil):
        """ 
        PIL 이미지를 입력받아 동기화된 
        (랜덤 크롭 + 랜덤 플립 + 텐서 변환)을 적용
        """
        
        # --- 1. To Tensor (0-255 -> 0-1) ---
        distorted_tensor = TF.to_tensor(distorted_img_pil)
        clean_tensor = TF.to_tensor(clean_img_pil)
        
        # --- 2. Random Crop (동기화) ---
        # 이미지가 패치 크기보다 작을 경우 패딩
        _, h, w = distorted_tensor.shape
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        if pad_h > 0 or pad_w > 0:
            # (top, bottom, left, right)
            padding = (0, 0, pad_w, pad_h) 
            distorted_tensor = TF.pad(distorted_tensor, padding=padding, fill=0)
            clean_tensor = TF.pad(clean_tensor, padding=padding, fill=0)

        i, j, h, w = T.RandomCrop.get_params(
            distorted_tensor, output_size=(self.patch_size, self.patch_size)
        )
        distorted_tensor = TF.crop(distorted_tensor, i, j, h, w)
        clean_tensor = TF.crop(clean_tensor, i, j, h, w)
        
        # --- 3. Random Flip (동기화) ---
        if random.random() > 0.5:
            distorted_tensor = TF.hflip(distorted_tensor)
            clean_tensor = TF.hflip(clean_tensor)
            
        return distorted_tensor, clean_tensor

    def __getitem__(self, idx):
        distorted_path, clean_path, dataset_name = self.image_pairs[idx]
        
        try:
            # 1. 이미지 로드 (PIL)
            # .convert('RGB')는 4채널 TIF나 1채널 흑백 이미지를 강제로 3채널 RGB로 통일
            distorted_img_pil = Image.open(distorted_path).convert('RGB')
            clean_img_pil = Image.open(clean_path).convert('RGB')
            
            # --- 2. Dual-Pipeline 로직 ---
            
            # 2a. 기본 변환 (Crop, Flip, ToTensor)
            # (Pipeline B도 이 정답 텐서가 필요하므로 먼저 수행)
            dist_tensor_orig, clean_tensor = self._get_transforms(distorted_img_pil, clean_img_pil)

            if random.random() < self.hybrid_prob:
                # --- [Pipeline B] - 실시간 하이브리드 합성 ---
                # (정답 이미지 'clean_tensor'를 기반으로 새 왜곡 생성)
                
                # 1~3개의 왜곡을 랜덤 순서로 적용
                hybrid_augs = [add_gaussian_blur, add_gaussian_noise, add_jpeg_compression]
                random.shuffle(hybrid_augs)
                num_augs = random.randint(1, 3) # 1개, 2개, 또는 3개 모두 적용
                
                new_distorted_tensor = clean_tensor
                for aug in hybrid_augs[:num_augs]:
                    new_distorted_tensor = aug(new_distorted_tensor)
                
                return new_distorted_tensor, clean_tensor

            else:
                # --- [Pipeline A] - 표준 쌍 사용 ---
                return dist_tensor_orig, clean_tensor

        except Exception as e:
            print(f"Error loading image: {distorted_path} (Dataset: {dataset_name})")
            print(f"Error details: {e}")
            # 훈련 중단 방지를 위해 더미 텐서 반환
            return torch.zeros(3, self.patch_size, self.patch_size), \
                   torch.zeros(3, self.patch_size, self.patch_size)

# -----------------------------------------------
#  테스트 코드
# -----------------------------------------------
if __name__ == "__main__":
    
    # 1. (필수) 자신의 데이터 경로 설정
    # (Windows 경로는 r'...' 또는 'G:/DPR-Net/data' 사용)
    DATA_DIR = r"G:\DPR-Net\data" 
    PATCH_SIZE = 128
    
    print(f"--- [Test] DPRDataset (Dual-Pipeline) ---")
    print(f"Target Data Directory: {DATA_DIR}")

    try:
        # 2. 'train' 모드 테스트 (Pipeline A + B)
        print("\n--- Testing 'train' mode (Hybrid Prob = 0.5) ---")
        train_dataset = DPRDataset(data_dir=DATA_DIR, mode='train', patch_size=PATCH_SIZE, hybrid_prob=0.5)
        
        if len(train_dataset) > 0:
            print(f"Found {len(train_dataset)} training samples.")
            # 5개 샘플 로드 테스트
            for i in range(min(5, len(train_dataset))):
                dist, clean = train_dataset[i] # __getitem__ 호출
                print(f"  Sample {i}: Dist={dist.shape}, Clean={clean.shape}, Dtype={dist.dtype}")
                assert dist.shape == (3, PATCH_SIZE, PATCH_SIZE)
                assert clean.shape == (3, PATCH_SIZE, PATCH_SIZE)
                assert dist.min() >= 0.0 and dist.max() <= 1.0

            print("\n✅ 'train' mode test passed.")
        else:
            print("⚠️  'train' dataset is EMPTY. Check paths and 'Train'/'our485' folder names.")

        
        # 3. 'test' 모드 테스트 (Pipeline A only)
        print("\n--- Testing 'test' mode (Hybrid Prob = 0.0) ---")
        test_dataset = DPRDataset(data_dir=DATA_DIR, mode='test', patch_size=PATCH_SIZE, hybrid_prob=0.0) # hybrid_prob=0

        if len(test_dataset) > 0:
            print(f"Found {len(test_dataset)} test samples.")
            dist_test, clean_test = test_dataset[0]
            print(f"  Test Sample 0: Dist={dist_test.shape}, Clean={clean_test.shape}")
            assert dist_test.shape == (3, PATCH_SIZE, PATCH_SIZE)
            print("\n✅ 'test' mode test passed.")
        else:
            print("⚠️  'test' dataset is EMPTY. Check paths and 'Test'/'eval15' folder names.")

    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("Please ensure 'DATA_DIR' variable is correct and folders exist.")
    except Exception as e:
        import traceback
        print(f"\n[Fatal Error] An unexpected error occurred:")
        traceback.print_exc()