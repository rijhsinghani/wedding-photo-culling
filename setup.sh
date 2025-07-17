#!/bin/bash

# Wedding Photo Culling Assistant - Setup Script for macOS/Linux
# This script handles the complete setup process

set -e  # Exit on error

echo "=== Wedding Photo Culling Assistant Setup ==="
echo "Platform: $(uname -s)"
echo "Python version: $(python3 --version)"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$PYTHON_VERSION < 3.9" | bc) -eq 1 ]]; then
    echo "Error: Python 3.9 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create required directories
echo "Creating required directories..."
mkdir -p models cache temp logs output
echo "✓ Directories created"

# Create virtual environment (optional but recommended)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
python3 -m pip install -r requirements.txt

# Download models
echo ""
echo "Downloading required models..."

MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

# Model URLs
declare -A MODELS=(
    ["haarcascade_frontalface_default.xml"]="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    ["haarcascade_frontalface_alt.xml"]="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml"
    ["haarcascade_eye.xml"]="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    ["deploy.prototxt"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    ["res10_300x300_ssd_iter_140000.caffemodel"]="https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    ["lbfmodel.yaml"]="https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    ["dino_model.pth"]="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
)

# Note: The open-closed-eye-model.h5 URL needs to be fixed - using a placeholder
MODELS["open-closed-eye-model.h5"]="https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"

# Download each model
for model in "${!MODELS[@]}"; do
    if [ ! -f "$MODEL_DIR/$model" ]; then
        echo "Downloading $model..."
        curl -L -o "$MODEL_DIR/$model" "${MODELS[$model]}" || echo "Warning: Failed to download $model"
    else
        echo "✓ $model already exists"
    fi
done

# Create default config.json if it doesn't exist
if [ ! -f "config.json" ]; then
    echo ""
    echo "Creating default configuration..."
    cat > config.json << 'EOF'
{
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
        "use_gpu": false,
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
EOF
    echo "✓ Created config.json"
fi

# Create .env file template if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env template..."
    cat > .env << 'EOF'
# Gemini API Configuration (optional - for AI-based quality assessment)
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your-api-key-here

# Processing Configuration
MAX_WORKERS=4
BATCH_SIZE=16
USE_GPU=false

# Logging
LOG_LEVEL=INFO
EOF
    echo "✓ Created .env template"
    echo "Note: Add your Gemini API key to .env if you want to use AI-based quality assessment"
fi

# Test imports
echo ""
echo "Testing Python imports..."
python3 -c "
import numpy
import cv2
import torch
import PIL
import rawpy
import transformers
import rich
print('✓ All core imports successful')
"

# Check CUDA availability
echo ""
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA is available: {torch.cuda.get_device_name(0)}')
else:
    print('ℹ CUDA not available - will use CPU')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "You can now run the application using:"
echo "  python3 cli.py"
echo ""
echo "For virtual environment setup, run:"
echo "  ./setup.sh --venv"
echo "  source venv/bin/activate"
echo "  python3 cli.py"
echo ""