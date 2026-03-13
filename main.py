import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("trl").setLevel(logging.ERROR)

import time
from pathlib import Path

from utils.ui_components import CyberpunkUI
from engine.hardware_monitor import HardwareManager
from engine.data_alchemist import DataAlchemist
from engine.trainer_core import TrainerCore

from rich.live import Live

def wait_for_files(ui):
    """Pauses until the user puts documents in data/raw."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    while True:
        valid_extensions = {".pdf", ".txt", ".csv", ".docx"}
        has_files = any(f.is_file() and f.suffix.lower() in valid_extensions for f in raw_dir.iterdir())
        if has_files:
            break
        ui.print_warning("No valid files found in data/raw. Please place .pdf, .txt, .csv, or .docx files there.")
        ui.print_info("Press Enter when ready, or Ctrl+C to abort.")
        input()
        
def main():
    ui = CyberpunkUI()
    ui.console.clear()
    
    # Initialize Phase 1
    hw_manager = HardwareManager()
    
    layout = ui.generate_layout(
        hw_manager.get_telemetry(), 
        phase_name="System Initialization"
    )
    ui.console.print(layout)
    time.sleep(2) # Give hardware sensors time to stabilize
    
    # 2. Wait for data
    ui.print_info(f"Targeting: {hw_manager.config['model_name']} ({hw_manager.config['mode']})")
    wait_for_files(ui)
    
    # Launch Data Processing
    alchemist = DataAlchemist(hw_manager)
    
    with Live(ui.generate_layout(hw_manager.get_telemetry(), phase_name="Data Synthesis"), refresh_per_second=4, screen=True) as live:
        def synthesis_progress(step, total, msg):
            live.update(ui.generate_layout(
                hw_manager.get_telemetry(),
                phase_name="Data Synthesis",
                msg=msg,
                synth_step=step,
                synth_total=total
            ))
            
        valid_pairs = alchemist.process_and_synthesize(progress_callback=synthesis_progress)
        
    ui.console.clear()
    if valid_pairs == 0:
        ui.print_warning("No valid training data could be generated. Aborting.")
        return
        
    ui.print_info(f"Successfully synthesized {valid_pairs} training pairs.")
    time.sleep(1)
    
    # 4. Training
    trainer_core = TrainerCore(hw_manager)
    trainer_core.setup_model()
    
    # We don't know total steps initially, the Trainer calculates it
    # We will pass a callback to TrainerCore
    
    def training_callback(step, total, loss):
        # We need a reference to `live` outside to update it during HF's train loop
        pass
        
    with Live(ui.generate_layout(hw_manager.get_telemetry(), phase_name="Model Alignment"), refresh_per_second=2, screen=True) as live:
        def update_train_ui(step, total, loss):
            if hw_manager.check_thermal_protection()[0]:
                time.sleep(10) # Pause silently to cool down without breaking Live UI
                
            live.update(ui.generate_layout(
                hw_manager.get_telemetry(), 
                step=step, 
                total_steps=total, 
                current_loss=loss,
                phase_name="Model Alignment"
            ))
            
        trainer_core.start_training(callback=update_train_ui)
        
    # 5. Export
    ui.console.clear()
    ui.console.print(ui.generate_layout(hw_manager.get_telemetry(), phase_name="Training Complete"))
    ui.print_info("Training Complete. Exporting LoRA weights...")
    trainer_core.save_and_export()
    ui.print_info("Export successful! The `inference.py` script has been generated.")
    
    # 6. Chat Interface
    if hasattr(trainer_core, "model") and hasattr(trainer_core, "tokenizer"):
        ui.chat_interface(model=trainer_core.model, tokenizer=trainer_core.tokenizer)
    else:
        ui.print_warning("Model or tokenizer not found. Chat interface unavailable.")

if __name__ == "__main__":
    main()
