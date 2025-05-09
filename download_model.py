import requests
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for data in response.iter_content(chunk_size=4096):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total_size)
                print(f"\rProgress: [{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')
    print("\nDownload complete!")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("video_super_resolution/models", exist_ok=True)
    
    # Download the model
    url = "https://huggingface.co/SherryX/STAR/resolve/main/I2VGen-XL-based/light_deg.pt"
    filename = "video_super_resolution/models/star.pth"
    download_file(url, filename) 