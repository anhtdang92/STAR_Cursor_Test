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
    # Create pretrained_weight directory if it doesn't exist
    os.makedirs("pretrained_weight", exist_ok=True)
    
    # Download the light degradation model
    url = "https://huggingface.co/SherryX/STAR/resolve/main/I2VGen-XL-based/light_deg.pt"
    filename = "pretrained_weight/light_deg.pt"
    download_file(url, filename)
    
    print("Model downloaded successfully!") 