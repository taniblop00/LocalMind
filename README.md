# 🧠 LocalMind — Autonomous Local Fine-Tuning Engine

> **Train your own AI model — on your own machine — from your own documents. Zero cloud, zero API keys, zero nonsense.**

LocalMind is a fully autonomous fine-tuning pipeline that takes **any document you have** (PDF, TXT, CSV, DOCX), synthesizes high-quality training data from it automatically, and fine-tunes a real language model on your hardware — all from a single command.

---

## 🤔 What Problem Does This Solve?

Most people who want a custom AI model face a brutal tradeoff:
- **Use a cloud API** → your data leaves your machine, costs money, and you don't own the model.
- **Fine-tune yourself** → complex setup, requires deep ML knowledge, different code for GPU vs CPU, etc.

**LocalMind eliminates that tradeoff.**

It detects your hardware automatically, picks the right model and settings for your specific machine, converts your documents into training data without any manual work, fine-tunes the model — and when it's done, gives you a ready-to-use `inference.py` script so you can chat with your new AI immediately.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Auto Hardware Detection** | Detects your GPU (VRAM) or CPU and picks the optimal model & settings |
| 🤖 **Intelligent Data Synthesis** | Uses NLP or a helper LLM to convert your documents into Q&A training pairs |
| 🎯 **LoRA Fine-Tuning** | Parameter-efficient training — works even on modest hardware |
| 🌡️ **Thermal Protection** | Monitors GPU temperature and pauses if overheating is detected |
| 💻 **Live Cyberpunk TUI** | A beautiful real-time terminal dashboard while training runs |
| 📦 **One-Click Export** | Exports LoRA weights + a standalone `inference.py` script automatically |
| 🖥️ **GPU + CPU Support** | Full support for NVIDIA GPUs via Unsloth, or CPU-only fallback |
| 📤 **Remote Upload Server** | Drag & drop files from your browser to a remote GPU server — no SSH needed |

---

## 🖥️ System Requirements

| | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 22.04 |
| **Python** | 3.10 | 3.11 |
| **RAM** | 8 GB | 16 GB+ |
| **GPU (optional)** | NVIDIA 6 GB VRAM | NVIDIA 8–16 GB VRAM |
| **Storage** | 20 GB free | 40 GB+ free |

> **No GPU? No problem.** LocalMind runs in CPU-only mode automatically, using a smaller model (`SmolLM-135M-Instruct`). It will be slower, but it works.

---

## 🚀 Quick Start — Step by Step

### Step 1 — Clone the Project

```bash
git clone https://github.com/taniblop00/LocalMind2.git
cd LocalMind2
```

### Step 2 — Run the Setup Script (One Time Only)

**On Windows:**
```batch
setup.bat
```

**On Linux / macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Check that Python 3.10+ is installed
2. Create a virtual environment (`venv/`) so nothing pollutes your system Python
3. Auto-detect if you have an NVIDIA GPU
4. Install the correct version of PyTorch (CUDA or CPU-only)
5. Install all dependencies from `requirements.txt`
6. Install Unsloth (GPU only, for fast training)
7. Download the spaCy language model for text parsing

> ⚠️ **This step can take 5–15 minutes** depending on your internet speed. PyTorch and the ML libraries are large.

---

### Step 3 — Add Your Documents

Place your files inside the `data/raw/` folder.

```
LocalMind2/
└── data/
    └── raw/          ← PUT YOUR FILES HERE
        ├── my_book.pdf
        ├── notes.txt
        ├── data.csv
        └── report.docx
```

**Supported formats:**
- `.pdf` — Books, papers, reports
- `.txt` — Plain text, logs, articles
- `.csv` — Tabular data, Q&A datasets
- `.docx` — Word documents

> The more content you provide, the better the model will be trained. A minimum of 1–2 pages of meaningful text is recommended.

---

### Step 4 — Activate the Virtual Environment

**Windows:**
```batch
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt. This means you are inside the isolated environment.

---

### Step 5 — Run the Application

```bash
python main.py
```

That's it. Everything from this point is automatic.

---

## 🔄 What Happens When You Run `main.py`

LocalMind runs through 5 automated phases, each shown in the live terminal dashboard:

```
Phase 1 → System Initialization   (Hardware scan, model selection)
Phase 2 → Data Synthesis          (Document ingestion + Q&A generation)
Phase 3 → Model Alignment         (Fine-tuning with LoRA)
Phase 4 → Training Complete       (Weights saved, inference script generated)
Phase 5 → Chat Interface          (Chat with your newly trained model)
```

### Phase 1 — System Initialization
The `HardwareManager` scans your system:
- Detects GPU model, VRAM, temperature
- Detects CPU physical/logical cores, RAM
- Selects the best model for your hardware:

| VRAM | Model Selected | LoRA Rank |
|---|---|---|
| No GPU (CPU only) | `SmolLM-135M-Instruct` | 4 |
| < 6 GB | `TinyLlama (4-bit)` | 8 |
| 6–14 GB | `Llama-3.2-3B-Instruct (4-bit)` | 16 |
| ≥ 14 GB | `Llama-3.1-8B (4-bit)` | 32 |

### Phase 2 — Data Synthesis
The `DataAlchemist` reads all files from `data/raw/` and:
1. Splits documents into semantic chunks with overlap
2. For each chunk, generates question-answer training pairs:
   - **GPU with ≥ 6GB VRAM**: Uses a separate 1B helper LLM (`Llama-3.2-1B`) that generates 3 Q&A pairs per chunk using different personas (Expert, Curious User, Skeptic)
   - **CPU / Low VRAM**: Uses spaCy NLP to extract key entities and form questions heuristically
3. Saves all pairs as Alpaca-format JSONL to `data/training_data.jsonl`

### Phase 3 — Model Fine-Tuning
`TrainerCore` loads your selected model with 4-bit quantization (GPU) and applies LoRA:
- Trains using the Hugging Face `SFTTrainer`
- Batch size, epochs, and gradient accumulation are calculated automatically based on your hardware
- Thermal protection: if GPU temperature exceeds 82°C, training pauses silently to cool down
- The live dashboard shows loss, step count, and hardware telemetry in real time

### Phase 4 — Export
When training is done:
- LoRA weights are saved to `outputs/lora_model/`
- On GPU: model is also exported in GGUF format (`q4_k_m`) to `outputs/model/`
- A standalone inference script is generated at `outputs/inference.py`

### Phase 5 — Chat Interface
After export, you can chat directly with your trained model inside the terminal.

---

## 💬 Using the Trained Model After Training

### Option A — Chat in the Same Session
At the end of `main.py`, a chat interface opens automatically. Type your message and press Enter.

### Option B — Run the Generated Inference Script

```bash
cd outputs
python inference.py
```

This script loads your LoRA weights on top of the base model and starts an interactive chat loop. Type `exit`, `quit`, or `q` to stop.

Example session:
```
Loading unsloth/Llama-3.2-3B-Instruct-bnb-4bit with NexusMind LoRA...
Model loaded successfully. Enter your prompt:
> What is the main topic of my documents?
[Model responds based on what it learned from your files]
```

---

## ✅ Verify Your Setup (Optional)

Before running `main.py`, you can run the verification script to make sure all components work correctly:

```bash
python verify_system.py
```

This runs 3 tests without downloading any large models:
1. **HardwareManager** — checks hardware detection and config selection
2. **CyberpunkUI** — validates the terminal dashboard rendering
3. **DataAlchemist** — runs the data synthesis pipeline on a small dummy file

Expected output:
```
=== NEXUSMIND SYSTEM VERIFICATION ===

[1/3] Testing Advanced HardwareManager (CPU Topologies)...
  System RAM: 6.2 / 15.9 GB
  CPU Cores (Physical): 8
  Selected Mode: gpu | Model: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
  [OK] HardwareManager initialized successfully.

[2/3] Testing CyberpunkUI Components & Live Lab...
  [OK] CyberpunkUI Live Lab and Celebration layouts generated without errors.

[3/3] Testing DataAlchemist Persona Synthesis...
  -> Streaming UI Message: [Curious User] What is the significance of NexusMind?
  [OK] DataAlchemist generated 1 valid pairs from dummy data.

=== VERIFICATION COMPLETE ===
```

---

## 🌐 Running on a Remote GPU Server (Vast.ai / RunPod / etc.)

### Step 1 — Clone & run setup

```bash
git clone https://github.com/taniblop00/LocalMind2.git
cd LocalMind2
chmod +x setup.sh && ./setup.sh
```

The setup script auto-detects your CUDA version and installs the exact matching PyTorch + torchvision — no manual version picking needed.

### Step 2 — Upload your training files

Run the built-in upload server:

```bash
source venv/bin/activate
python upload_server.py
```

Then open the URL printed in the terminal (e.g. `http://123.45.67.89:7860`) in your browser.
Drag & drop your `.pdf`, `.txt`, `.csv`, or `.docx` files — they upload directly to `data/raw/` with a progress bar.

> **Vast.ai:** In your instance dashboard → **Open Ports** → expose port `7860`, then use the link shown.
>
> **RunPod:** Connect → HTTP Service → Port `7860`.

Press **Ctrl+C** when done uploading.

### Step 3 — Run training

```bash
python main.py
```

---

## 📂 Project Structure

```
LocalMind2/
├── main.py                 # Entry point — runs the full pipeline
├── upload_server.py        # Remote file uploader (browser drag & drop)
├── verify_system.py        # Optional system verification script
├── requirements.txt        # Python dependencies
├── setup.bat               # Windows setup script
├── setup.sh                # Linux/macOS setup script
│
├── engine/
│   ├── hardware_monitor.py # Hardware detection & telemetry
│   ├── data_alchemist.py   # Document ingestion & Q&A synthesis
│   └── trainer_core.py     # LoRA fine-tuning & model export
│
├── utils/
│   └── ui_components.py    # Cyberpunk terminal UI (Rich-based)
│
├── data/
│   ├── raw/                # ← PUT YOUR INPUT DOCUMENTS HERE
│   ├── processed/          # Intermediate processed files
│   └── training_data.jsonl # Generated training pairs (auto-created)
│
└── outputs/
    ├── lora_model/         # LoRA adapter weights (auto-created)
    ├── model/              # GGUF model (GPU only, auto-created)
    └── inference.py        # Standalone chat script (auto-created)
```

---

## ❓ Common Issues & Fixes

**`Python is not installed or not in PATH`**
→ Install Python 3.10 or 3.11 from [python.org](https://www.python.org/downloads/). During install, check **"Add Python to PATH"**.

**`No valid files found in data/raw`**
→ Make sure your files are inside the `data/raw/` folder and have one of the supported extensions (`.pdf`, `.txt`, `.csv`, `.docx`).

**`No valid training data could be generated`**
→ Your documents may be too short or in an unsupported encoding. Try converting them to `.txt` first and ensure the content is in English.

**Training is very slow**
→ CPU-only mode is expected to be slow. For faster training, use a machine with an NVIDIA GPU and re-run `setup.bat`/`setup.sh` to install CUDA dependencies.

**Out of memory during training**
→ Close other applications to free RAM/VRAM and try again. On GPU, the system already uses 4-bit quantization to minimize memory usage.

**`[ERROR] GPU Initialization Failed`**
→ This is non-fatal. LocalMind will automatically fall back to CPU mode. GPU errors usually mean an outdated driver — update your NVIDIA drivers from [nvidia.com](https://www.nvidia.com/drivers).

---

## 📋 Full Dependency List

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.3.1 | Core deep learning framework |
| `transformers` | 4.43.3 | HuggingFace model loading |
| `peft` | 0.12.0 | LoRA adapter implementation |
| `trl` | 0.9.6 | SFT (Supervised Fine-Tuning) Trainer |
| `accelerate` | 0.33.0 | Multi-device training utilities |
| `datasets` | 3.0.0 | Dataset loading and processing |
| `bitsandbytes` | 0.43.3 | 4-bit / 8-bit quantization |
| `unsloth` | latest | Fast GPU training (auto-installed) |
| `langchain-community` | 0.2.16 | Document loaders (PDF, DOCX, CSV) |
| `spacy` | 3.8.0 | NLP for heuristic Q&A generation |
| `rich` | 13.7.1 | Terminal UI / dashboard rendering |
| `pynvml` | 11.5.0 | NVIDIA GPU monitoring |
| `sentencepiece` | 0.2.0 | Tokenizer support |
| `docx2txt` | latest | DOCX file reading |

---

## 👤 Author

Built by **taniblop00**

---

*LocalMind — Because your data should stay yours.*
