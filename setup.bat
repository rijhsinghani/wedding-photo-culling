@echo off
echo Creating virtual environment...
python -m venv env
call env\Scripts\activate

echo Installing PyTorch with CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo Installing other requirements...
pip install -r requirements.txt

echo Setting up InsightFace...
python -c "import insightface; insightface.utils.prepare_facebank()"

echo Setup complete!
pause