import os
import subprocess
import sys
import platform
import urllib.request
import shutil
from pathlib import Path

def run_command(command, shell=True):
    """Run a shell command and print its output."""
    process = subprocess.Popen(
        command,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"Downloading {url} to {destination}")
    urllib.request.urlretrieve(url, destination)

def setup_environment():
    """Set up the Python virtual environment and install dependencies."""
    print("Setting up virtual environment...")
    
    # Create virtual environment
    if platform.system() == "Windows":
        run_command("python -m venv venv")
        activate_script = "venv\\Scripts\\activate"
    else:
        run_command("python3 -m venv venv")
        activate_script = "source venv/bin/activate"
    
    # Install dependencies
    print("Installing dependencies...")
    if platform.system() == "Windows":
        run_command(f"{activate_script} && pip install -r requirements.txt")
    else:
        run_command(f"{activate_script} && pip install -r requirements.txt")

def download_models():
    """Download required model files."""
    print("Downloading model files...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download model files
    model_urls = {
        "star_diffusion_model.pth": "https://huggingface.co/your-model-repo/resolve/main/star_diffusion_model.pth",
        "star_diffusion_config.json": "https://huggingface.co/your-model-repo/resolve/main/star_diffusion_config.json"
    }
    
    for filename, url in model_urls.items():
        destination = os.path.join("models", filename)
        if not os.path.exists(destination):
            download_file(url, destination)

def setup_backend():
    """Set up the backend server."""
    print("Setting up backend server...")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Create a test video if it doesn't exist
    if not os.path.exists("test_video.mp4"):
        print("Creating test video...")
        # You can add code here to create a test video if needed
        pass

def main():
    """Main setup function."""
    print("Starting setup process...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run setup steps
    setup_environment()
    download_models()
    setup_backend()
    
    print("\nSetup completed successfully!")
    print("\nTo start the backend server, run:")
    if platform.system() == "Windows":
        print("venv\\Scripts\\python backend\\server.py")
    else:
        print("source venv/bin/activate && python backend/server.py")

if __name__ == "__main__":
    main() 