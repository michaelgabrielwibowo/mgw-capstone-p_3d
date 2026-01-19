"""
SAM Model Download Script
Downloads the SAM ViT-B checkpoint from Meta's repository.
"""
import os
import requests
from tqdm import tqdm
import config

def download_sam_model():
    """
    Downloads the SAM ViT-B model weights if they don't already exist.
    """
    model_path = config.SAM_MODEL_PATH
    model_name = os.path.basename(model_path)
    
    if os.path.exists(model_path):
        print(f"Found existing SAM model: {model_name}")
        file_size = os.path.getsize(model_path)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        return model_path
    
    print(f"{model_name} not found. Downloading...")
    
    # SAM ViT-B checkpoint URL
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f, tqdm(
            desc=model_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                bar.update(size)
        
        print(f"\nSAM model downloaded successfully to: {model_path}")
        return model_path
        
    except requests.exceptions.RequestException as e:
        print(f"\nFailed to download the model: {e}")
        return None

if __name__ == "__main__":
    download_sam_model()
