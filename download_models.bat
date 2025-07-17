@echo off
echo Setting up environment for photo culling system...

echo Creating necessary directories...
if not exist "models" mkdir models
if not exist "temp" mkdir temp
if not exist "output" mkdir output
if not exist "cache" mkdir cache
if not exist "logs" mkdir logs

echo Installing basic requirements...
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo Installing accelerate and other requirements...
pip install "accelerate>=0.26.0"
pip install -r requirements.txt

echo Installing InsightFace...
pip uninstall insightface onnxruntime -y
pip install insightface==0.7.3 onnxruntime

echo Downloading required models...
python -c "from models import download_required_models; download_required_models()"

echo Setting up face detection models...
python -c "import insightface; app = insightface.app.FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'], providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1)"

echo Setting up quality assessment models...
python -c "import torch; from transformers import AutoModelForImageClassification, AutoImageProcessor; model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-50', cache_dir='models'); processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50', cache_dir='models')"

echo Checking system configuration...
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); import onnxruntime as ort; print(f'ONNX Runtime Providers: {ort.get_available_providers()}')"

echo Testing system initialization...
python -c "from app import PhotoAnalyzer; import json; config = json.load(open('config.json')); analyzer = PhotoAnalyzer(config); print('System initialized successfully')"

echo Installation completed! Press any key to continue...
pause
