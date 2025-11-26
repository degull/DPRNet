import os

# 프로젝트 루트 경로 (사용자 환경에 맞춤)
ROOT_DIR = r"G:\DPR-Net"

# 생성할 디렉토리 목록
DIRS = [
    r"configs",
    r"data",
    r"data\CSD",
    r"data\lol_dataset",
    r"data\rain100H",
    r"data\SOTS",
    r"logs",
    r"models",
    r"preprocessing",
    r"utils",
]

# 생성할 빈 파일 목록 (패키지 인식용 __init__.py 포함)
FILES = [
    r"configs\dpr_config.yaml",
    
    r"data\__init__.py",
    r"data\dataset.py",
    
    r"models\__init__.py",
    r"models\clip_encoder.py",
    r"models\mistral_llm.py",
    r"models\pixel_decoder.py",
    r"models\film_layer.py",
    r"models\vetnet.py",
    r"models\dpr_net_v2.py",
    
    r"preprocessing\preprocess_captions.py",
    
    r"utils\__init__.py",
    r"utils\visualization.py",
    
    r".gitignore",
    r"train.py",
    r"inference.py",
    r"requirements.txt"
]

def create_structure():
    print(f"🚀 Creating project structure at: {ROOT_DIR}")

    # 1. 루트 디렉토리 생성
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    # 2. 하위 폴더 생성
    for dir_name in DIRS:
        dir_path = os.path.join(ROOT_DIR, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"   [Folder] Created: {dir_path}")
        else:
            print(f"   [Folder] Exists: {dir_path}")

    # 3. 파일 생성
    for file_name in FILES:
        file_path = os.path.join(ROOT_DIR, file_name)
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                pass # 빈 파일 생성
            print(f"   [File]   Created: {file_path}")
        else:
            print(f"   [File]   Exists: {file_path}")

    print("\n✅ Project structure setup complete!")

if __name__ == "__main__":
    create_structure()