import os
import shutil
import time
from pathlib import Path

def setup_dummy_data():
    """Creates a temporary dummy text file for DataAlchemist to process."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    with open(raw_dir / "dummy_test.txt", "w", encoding="utf-8") as f:
        f.write("NexusMind is an advanced autonomous system for generating JSON datasets from messy text. It uses dynamic quantization and persona-based Alchemists.")
        
def cleanup_dummy_data():
    """Removes the temporary data folder."""
    if os.path.exists("data"):
        shutil.rmtree("data")

print("=== NEXUSMIND SYSTEM VERIFICATION ===")

# Test 1: HardwareManager Initialization
print("\n[1/3] Testing Advanced HardwareManager (CPU Topologies)...")
try:
    from engine.hardware_monitor import HardwareManager
    hw = HardwareManager()
    cfg = hw.get_optimal_config()
    telemetry = hw.get_telemetry()
    print(f"  System RAM: {telemetry['ram_used_gb']:.1f} / {telemetry['ram_total_gb']:.1f} GB")
    print(f"  CPU Cores (Physical): {hw.cpu_physical_cores}")
    print(f"  CPU Cores (Logical): {hw.cpu_logical_cores}")
    print(f"  Threads Assigned to DataLoader: {cfg.get('dataloader_workers', 1)}")
    if hw.has_gpu:
        print(f"  GPU Detected: {hw.gpu_name} (VRAM: {telemetry['vram_total_gb']:.1f} GB)")
    print(f"  Selected Mode: {cfg['mode']} | Model: {cfg['model_name']}")
    print("  [OK] HardwareManager initialized successfully.")
except Exception as e:
    print(f"  [FAIL] HardwareManager error: {e}")

# Test 2: UI Rendering (Live Lab Headless)
print("\n[2/3] Testing CyberpunkUI Components & Live Lab...")
try:
    from utils.ui_components import CyberpunkUI
    ui = CyberpunkUI()
    layout = ui.generate_layout(hw.get_telemetry(), phase_name="Data Synthesis", msg="[Expert] Validating generation pipeline.")
    layout_celebration = ui.generate_layout(hw.get_telemetry(), phase_name="Training Complete")
    print("  [OK] CyberpunkUI Live Lab and Celebration layouts generated without errors.")
except Exception as e:
    print(f"  [FAIL] CyberpunkUI error: {e}")

# Test 3: DataAlchemist (Persona/Heuristic pipeline)
print("\n[3/3] Testing DataAlchemist Persona Synthesis...")
try:
    from engine.data_alchemist import DataAlchemist
    setup_dummy_data()
    
    # Force heuristic mode to test extraction without massive 1B download
    alchemist = DataAlchemist(hw)
    alchemist.use_llm_worker = False 
    
    def log_progress(step, total, msg):
        print(f"  -> Streaming UI Message: {msg}")
        
    pairs_generated = alchemist.process_and_synthesize(progress_callback=log_progress)
    
    print(f"  [OK] DataAlchemist generated {pairs_generated} valid pairs from dummy data.")
    
    cleanup_dummy_data()
except Exception as e:
    print(f"  [FAIL] DataAlchemist error: {e}")

print("\n=== VERIFICATION COMPLETE ===")
print("Run `python main.py` to start the actual application flow.")
