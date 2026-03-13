#!/bin/bash
set -e

echo "===================================================="
echo "   NEXUSMIND - AUTONOMOUS FINE-TUNING ENGINE SETUP"
echo "===================================================="

echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed."
    exit 1
fi

echo "[2/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "[3/5] Detecting CUDA Availability..."
HAS_CUDA=0
if command -v nvcc &> /dev/null; then
    HAS_CUDA=1
    echo "[INFO] NVIDIA CUDA toolkit detected via nvcc."
elif command -v nvidia-smi &> /dev/null; then
    HAS_CUDA=1
    echo "[INFO] NVIDIA GPU detected via nvidia-smi."
else
    echo "[INFO] No NVIDIA GPU detected. Proceeding with CPU-Optimized installation."
fi

echo "[4/5] Installing core dependencies..."
if [ $HAS_CUDA -eq 1 ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio
    echo "Installing Xformers..."
    pip install xformers
    echo "Installing core requirements..."
    pip install -r requirements.txt
    echo "Installing Unsloth..."
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
else
    echo "Installing PyTorch (CPU Only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo "Installing core requirements (Skipping unsloth/xformers for safety)..."
    pip install -r requirements.txt
fi

echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "===================================================="
echo "[5/5] Setup Complete!"
echo "Run the application using:"
echo "source venv/bin/activate && python main.py"
echo "===================================================="
