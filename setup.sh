#!/bin/bash
set -e

echo "===================================================="
echo "   LOCALMIND - AUTONOMOUS FINE-TUNING ENGINE SETUP"
echo "===================================================="

# ── 1. Python check ──────────────────────────────────────
echo ""
echo "[1/5] Checking Python installation..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python is not installed. Please install Python 3.10 or 3.11."
    exit 1
fi
$PYTHON_CMD --version
echo "[OK] Python found."

# ── 2. Virtual environment ───────────────────────────────
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "[OK] Virtual environment created."
else
    echo "[OK] Virtual environment already exists."
fi
source venv/bin/activate

# Upgrade pip silently
pip install --upgrade pip --quiet

# ── 3. Detect CUDA version ───────────────────────────────
echo ""
echo "[3/5] Detecting hardware..."
HAS_CUDA=0
CUDA_MAJOR=0

if command -v nvidia-smi &> /dev/null; then
    HAS_CUDA=1
    # Extract major CUDA version from nvidia-smi (e.g. "12.8" → "12")
    CUDA_VERSION_RAW=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    CUDA_MAJOR=$(echo "$CUDA_VERSION_RAW" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION_RAW" | cut -d. -f2)
    echo "[INFO] GPU detected. CUDA Version: ${CUDA_VERSION_RAW}"
else
    echo "[INFO] No NVIDIA GPU detected. Using CPU-only mode."
fi

# ── 4. Install PyTorch (CUDA version–matched) ─────────────
echo ""
echo "[4/5] Installing dependencies..."

if [ $HAS_CUDA -eq 1 ]; then
    # Map CUDA version to PyTorch wheel tag
    if [ "$CUDA_MAJOR" -ge 13 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "${CUDA_MINOR:-0}" -ge 6 ]); then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        WHEEL_TAG="cu128"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "${CUDA_MINOR:-0}" -ge 3 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        WHEEL_TAG="cu124"
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        WHEEL_TAG="cu121"
    else
        # CUDA 11.x
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        WHEEL_TAG="cu118"
    fi

    echo "[INFO] Installing PyTorch for CUDA ${WHEEL_TAG}..."
    # Install torch, torchvision, torchaudio from the SAME index (prevents version mismatch)
    pip install torch torchvision torchaudio \
        --index-url "$TORCH_INDEX" \
        --quiet

    echo "[INFO] Installing project dependencies..."
    pip install -r requirements.txt --quiet

    echo "[INFO] Installing Unsloth (GPU-accelerated training)..."
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet || \
        echo "[WARN] Unsloth install failed — will fall back to standard HuggingFace training."

else
    echo "[INFO] Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
        --quiet

    echo "[INFO] Installing project dependencies..."
    pip install -r requirements.txt --quiet
fi

# ── 5. spaCy model ───────────────────────────────────────
echo ""
echo "[5/5] Downloading language model..."
$PYTHON_CMD -m spacy download en_core_web_sm --quiet

echo ""
echo "===================================================="
echo " Setup Complete!"
echo "===================================================="
echo ""
echo " To start training:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo " To upload files (remote server):"
echo "   Use JupyterLab — open the file browser and"
echo "   drag your files into the data/raw/ folder."
echo "===================================================="
