@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo    NEXUSMIND - AUTONOMOUS FINE-TUNING ENGINE SETUP
echo ====================================================

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. 
    echo Please install Python 3.10 or 3.11 and try again.
    pause
    exit /b 1
)

echo [2/5] Creating virtual environment...
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [3/5] Detecting CUDA Availability...
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] nvcc not found in PATH. Checking via nvidia-smi...
    nvidia-smi >nul 2>&1
    if !errorlevel! neq 0 (
        set HAS_CUDA=0
        echo [INFO] No NVIDIA GPU detected. Proceeding with CPU-Optimized installation.
    ) else (
        set HAS_CUDA=1
        echo [INFO] NVIDIA GPU detected via nvidia-smi. Proceeding with GPU (CUDA) installation.
    )
) else (
    set HAS_CUDA=1
    echo [INFO] NVIDIA CUDA toolkit detected. Proceeding with GPU (CUDA) installation.
)

echo [4/5] Installing core dependencies...
if !HAS_CUDA!==1 (
    echo Installing PyTorch with CUDA 12.1 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo Installing Xformers (required for Unsloth)...
    pip install xformers
    echo Installing base requirements...
    pip install -r requirements.txt
    echo Installing Unsloth...
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
) else (
    echo Installing PyTorch (CPU Only)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo Installing base requirements (Skipping unsloth/xformers)...
    pip install -r requirements.txt
)

echo Downloading spaCy model for CPU fallback parsing...
python -m spacy download en_core_web_sm

echo ====================================================
echo [5/5] Setup Complete!
echo Run the application using:
echo venv\Scripts\activate
echo python main.py
echo ====================================================
pause
