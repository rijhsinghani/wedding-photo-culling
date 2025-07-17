import os
import sys
import ssl
import urllib.request
import subprocess
import bz2
from pathlib import Path
import certifi
from tqdm import tqdm
import json
import platform
import shutil

class SetupPhotoProcessor:
    """Setup class for Photo Culling Assistant."""
    
    REQUIRED_DIRS = ["models", "cache", "temp", "logs", "output"]
    
    MODEL_URLS = {
            'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml':'https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/haarcascade_frontalface_alt.xml',
            'haarcascade_eye.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml',
            'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
            'res10_300x300_ssd_iter_140000.caffemodel': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
            'lbfmodel.yaml': 'https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml',
            'open-closed-eye-model.h5': 'https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5',
            'dino_model.pth': 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth'
        }
    
    def __init__(self):
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.platform = platform.system()
        self.python_version = sys.version_info
        print(f"Setting up for platform: {self.platform}")
        print(f"Python version: {self.python_version.major}.{self.python_version.minor}")

    def create_directories(self):
        """Create necessary directories."""
        print("\nCreating required directories...")
        for directory in self.REQUIRED_DIRS:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")

    def download_file(self, url: str, destination: str, desc: str):
        """Download a file with progress bar."""
        try:
            with urllib.request.urlopen(url, context=self.ssl_context) as response:
                total_size = int(response.headers['Content-Length'])
                
                with open(destination, 'wb') as f, tqdm(
                    desc=f"Downloading {desc}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    while True:
                        data = response.read(8192)
                        if not data:
                            break
                        size = f.write(data)
                        pbar.update(size)
            return True
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False

    def verify_python_version(self):
        """Verify Python version is 3.9.x or higher"""
        if not (self.python_version.major == 3 and self.python_version.minor >= 9):
            print(f"⚠️ Warning: This setup requires Python 3.9 or higher")
            print(f"Current Python version: {self.python_version.major}.{self.python_version.minor}")
            return False
        return True

    def install_requirements(self):
        """Install required packages."""
        print("\nInstalling required packages...")
        
        if not self.verify_python_version():
            print("Continuing with current Python version...")
            # Don't return False - allow setup to continue

        try:
            # Update pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # Install PyTorch first
            print("\nInstalling PyTorch...")
            torch_command = [
                sys.executable, "-m", "pip", "install",
                "torch", 
                "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu117"
            ]
            subprocess.run(torch_command, check=True)
            
            # Install core dependencies
            print("\nInstalling core dependencies...")
            core_packages = [
                "numpy<2.0",  # Compatible with Python 3.10
                "opencv-python",
                "Pillow",
                "rich",
                "colorama",
                "tqdm",
                "psutil",
                "rawpy",
                "scikit-image",
                "imagededup",
                "transformers"
            ]
            
            for package in core_packages:
                print(f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            
            # Install remaining requirements
            print("\nInstalling remaining requirements...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            
            print("\n✓ Successfully installed required packages")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error installing requirements: {str(e)}")
            return False
        except Exception as e:
            print(f"\n✗ Unexpected error: {str(e)}")
            return False

    def verify_torch_cuda(self):
        """Verify PyTorch CUDA installation."""
        print("\nChecking PyTorch CUDA support...")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
                print(f"✓ PyTorch CUDA version: {torch.version.cuda}")
            else:
                print("ℹ CUDA is not available. Using CPU mode.")
        except ImportError:
            print("✗ PyTorch not installed properly")

    def verify_installation(self):
        """Verify all components are installed correctly."""
        print("\nVerifying installation...")
        required_packages = [
            'numpy', 'cv2', 'torch', 'PIL', 'imagededup', 
            'rich', 'tqdm', 'rawpy', 'transformers'
        ]
        
        success = True
        print("\nChecking required packages:")
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package}")
            except ImportError:
                print(f"✗ {package} not installed")
                success = False
        
        return success

    def create_config(self):
        """Create default configuration file."""
        print("\nCreating configuration file...")
        config = {
            "paths": {
                "models_directory": "models",
                "cache_directory": "cache",
                "temp_directory": "temp",
                "log_directory": "logs",
                "output_directory": "output"
            },
            "thresholds": {
                "blur_threshold": 25,
                "quality_threshold": 0.7,
                "resolution_min": 300,
                "contrast_min": 80,
                "sharpness_min": 500,
                "fft_size": 60,
                "fft_thresh": 10,
                "focus_threshold": 50,
                "off_focus_threshold": 20.0,
                "eye_confidence": 50,
                "duplicate_hash_threshold": 50
            },
            "processing_settings": {
                "batch_size": 16,
                "max_workers": 4,
                "use_gpu": True,
                "memory_limit": 8
            },
            "supported_formats": {
                "standard": [
                    ".jpg", ".JPG",
                    ".jpeg", ".JPEG",
                    ".png", ".PNG",
                    ".bmp", ".BMP",
                    ".tiff", ".TIFF"
                ],
                "raw": [
                    ".cr2", ".CR2",
                    ".nef", ".NEF",
                    ".arw", ".ARW",
                    ".raf", ".RAF",
                    ".orf", ".ORF"
                ]
            }
        }
        
        if not os.path.exists('config.json'):
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=4)
            print("✓ Created config.json")
        else:
            print("ℹ config.json already exists")

    def download_models(self):
        """Download required model files."""
        print("\nDownloading required models...")
        models_path = Path('models')
        models_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, url in self.MODEL_URLS.items():
            destination = models_path / model_name
            if not destination.exists():
                print(f"\nDownloading {model_name}...")
                success = self.download_file(url, str(destination), model_name)
                if success:
                    print(f"✓ Downloaded {model_name}")
                    # Removed the process_downloaded_file call
                else:
                    print(f"✗ Failed to download {model_name}")
            else:
                print(f"✓ {model_name} already exists")

    def run(self):
        """Run the complete setup process."""
        print("\n=== Wedding Photo Culling Assistant Setup ===\n")
        
        try:
            if not self.verify_python_version():
                return False
                
            self.create_directories()
            install_success = self.install_requirements()
            
            if not install_success:
                print("\n⚠️ Some requirements could not be installed.")
                print("The application may still work with limited functionality.")
            
            self.download_models()
            self.verify_torch_cuda()
            self.create_config()
            
            if self.verify_installation():
                print("\n✓ Setup completed successfully!")
                print("\nYou can now run the application using: python cli.py")
                return True
            else:
                print("\n⚠️ Setup completed with warnings.")
                print("Some features might be limited.")
                print("\nYou can still run the application using: python cli.py")
                return True
                
        except Exception as e:
            print(f"\n✗ Error during setup: {str(e)}")
            return False

if __name__ == "__main__":
    setup = SetupPhotoProcessor()
    if not setup.run():
        sys.exit(1)
