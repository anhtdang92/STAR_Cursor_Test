import os
import requests
from tqdm import tqdm
from pathlib import Path

def download_file(url: str, destination: str, chunk_size: int = 8192) -> None:
    """Download a file from a URL to a destination path."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress bar
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)

def main():
    # Model URL - updated to use the correct model file
    model_url = "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors"
    
    # Model path
    model_path = Path(__file__).parent.parent / "models" / "star.safetensors"
    
    print(f"Downloading model to {model_path}...")
    download_file(model_url, str(model_path))
    print("Download complete!")

if __name__ == '__main__':
    main() 