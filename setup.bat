@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo    LOCALMIND - AUTONOMOUS FINE-TUNING ENGINE SETUP
echo ====================================================

:: ── 1. Python check ──────────────────────────────────────
echo.
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or 3.11 from python.org
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found.

:: ── 2. Virtual environment ───────────────────────────────
echo.
echo [2/5] Setting up virtual environment...
if not exist venv (
    python -m venv venv
    echo [OK] Virtual environment created.
) else (
    echo [OK] Already exists.
)
call venv\Scripts\activate.bat
pip install --upgrade pip --quiet

:: ── 3. Detect CUDA version ───────────────────────────────
echo.
echo [3/5] Detecting hardware...
set HAS_CUDA=0
set CUDA_MAJOR=0
set TORCH_INDEX=https://download.pytorch.org/whl/cpu

nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set HAS_CUDA=1
    echo [INFO] NVIDIA GPU detected.

    :: Extract CUDA major version from nvidia-smi output
    for /f "tokens=9 delims= " %%a in ('nvidia-smi ^| findstr "CUDA Version"') do (
        set CUDA_VER=%%a
    )
    :: Get major version (first digit before the dot)
    for /f "delims=." %%a in ("!CUDA_VER!") do set CUDA_MAJOR=%%a

    echo [INFO] CUDA Version detected: !CUDA_VER! (Major: !CUDA_MAJOR!)

    if !CUDA_MAJOR! GEQ 13 (
        set TORCH_INDEX=https://download.pytorch.org/whl/cu128
        echo [INFO] Using CUDA 12.8+ wheel ^(cu128^).
    ) else if !CUDA_MAJOR! EQU 12 (
        set TORCH_INDEX=https://download.pytorch.org/whl/cu128
        echo [INFO] Using CUDA 12.x wheel ^(cu128^).
    ) else (
        set TORCH_INDEX=https://download.pytorch.org/whl/cu118
        echo [INFO] Using CUDA 11.x wheel ^(cu118^).
    )
) else (
    echo [INFO] No NVIDIA GPU detected. Proceeding with CPU-only install.
)

:: ── 4. Install PyTorch (version-matched) ─────────────────
echo.
echo [4/5] Installing dependencies...
echo [INFO] Installing PyTorch, torchvision, torchaudio from: !TORCH_INDEX!
pip install torch torchvision torchaudio --index-url !TORCH_INDEX! --quiet

echo [INFO] Installing project dependencies...
pip install -r requirements.txt --quiet

if !HAS_CUDA! EQU 1 (
    echo [INFO] Installing Unsloth for GPU-accelerated training...
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet
    if %errorlevel% neq 0 (
        echo [WARN] Unsloth install failed. Will use standard HuggingFace training instead.
    )
)

:: ── 5. spaCy model ───────────────────────────────────────
echo.
echo [5/5] Downloading language model...
python -m spacy download en_core_web_sm --quiet

echo.
echo ====================================================
echo  Setup Complete!
echo ====================================================
echo.
echo  To start training:
echo    venv\Scripts\activate
echo    python main.py
echo ====================================================
pause
