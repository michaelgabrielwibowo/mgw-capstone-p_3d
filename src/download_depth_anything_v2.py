import os
import torch
import requests
from tqdm import tqdm
import config

def download_depth_anything_v2_weights():
    """
    Downloads the Depth Anything V2 model weights if they don't already exist.
    """
    model_name = os.path.basename(config.DEPTH_MODEL_PATH)
    model_path = config.DEPTH_MODEL_PATH
    
    if os.path.exists(model_path):
        print(f"Found existing model file: {model_name}")
        file_size = os.path.getsize(model_path)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            print("Model file is valid and can be loaded.")
            return model_path
        except Exception as e:
            print(f"Error loading model file: {e}")
            print("The existing model file might be corrupted. Please delete it and run the script again.")
            return None
            
    print(f"{model_name} not found. Downloading...")
    
    url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
    
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
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
                
        print(f"\nModel downloaded successfully to: {model_path}")
        return model_path
        
    except requests.exceptions.RequestException as e:
        print(f"\nFailed to download the model: {e}")
        return None

if __name__ == "__main__":
    download_depth_anything_v2_weights()