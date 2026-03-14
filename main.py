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


def wait_for_files(ui: CyberpunkUI, hw_manager: HardwareManager):
    """Pause until the user places documents in data/raw."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".pdf", ".txt", ".csv", ".docx"}

    while True:
        has_files = any(
            f.is_file() and f.suffix.lower() in valid_extensions
            for f in raw_dir.iterdir()
        )
        if has_files:
            break
        ui.print_warning(
            "No valid files found in data/raw — "
            "please place .pdf, .txt, .csv, or .docx files there."
        )
        ui.print_info("Press Enter when ready, or Ctrl+C to abort.")
        input()


def main():
    ui = CyberpunkUI()
    ui.console.clear()

    # ── Phase 1: System Init ──────────────────────────────────────────────
    hw_manager = HardwareManager()
    ui.set_phase("System Initialization")

    with Live(
        ui.generate_layout(hw_manager.get_telemetry()),
        console=ui.console,
        refresh_per_second=2,   # stable: renders at most 2×/s
        screen=True,
        transient=False,
    ) as live:
        # show hardware for 2 seconds so sensors can stabilise
        for _ in range(4):
            time.sleep(0.5)
            live.update(ui.generate_layout(hw_manager.get_telemetry()))

    ui.console.clear()
    ui.print_info(
        f"Target model: {hw_manager.config['model_name']}  "
        f"({hw_manager.config['mode'].upper()} mode)"
    )
    wait_for_files(ui, hw_manager)

    # ── Phase 2: Data Synthesis ───────────────────────────────────────────
    alchemist = DataAlchemist(hw_manager)
    ui.set_phase("Data Synthesis")

    with Live(
        ui.generate_layout(hw_manager.get_telemetry()),
        console=ui.console,
        refresh_per_second=4,
        screen=True,
        transient=False,
    ) as live:
        def synthesis_progress(step, total, msg):
            ui.push_synth_msg(step, total, msg)
            live.update(ui.generate_layout(hw_manager.get_telemetry()))

        valid_pairs = alchemist.process_and_synthesize(
            progress_callback=synthesis_progress
        )

    ui.console.clear()
    if valid_pairs == 0:
        ui.print_warning("No valid training data could be generated — aborting.")
        return

    ui.print_info(f"Synthesis complete: {valid_pairs} training pairs written.")
    time.sleep(1)

    # ── Phase 3: Model Setup ──────────────────────────────────────────────
    trainer_core = TrainerCore(hw_manager)
    trainer_core.setup_model()

    # ── Phase 4: Training ─────────────────────────────────────────────────
    ui.set_phase("Model Alignment")

    with Live(
        ui.generate_layout(hw_manager.get_telemetry()),
        console=ui.console,
        refresh_per_second=2,
        screen=True,
        transient=False,
    ) as live:
        def update_train_ui(step, total, loss):
            # Thermal protection: pause without breaking live UI
            if hw_manager.check_thermal_protection()[0]:
                time.sleep(10)

            ui.push_train_update(step, total, loss)
            live.update(ui.generate_layout(hw_manager.get_telemetry()))

        trainer_core.start_training(callback=update_train_ui)

    # ── Phase 5: Export ───────────────────────────────────────────────────
    ui.console.clear()
    ui.set_phase("Training Complete")
    ui.console.print(ui.generate_layout(hw_manager.get_telemetry()))

    ui.print_info("Training complete — exporting LoRA weights…")
    trainer_core.save_and_export()
    ui.print_info("Export successful! inference.py has been generated.")

    # ── Phase 6: Chat REPL ────────────────────────────────────────────────
    if hasattr(trainer_core, "model") and hasattr(trainer_core, "tokenizer"):
        ui.chat_interface(
            model=trainer_core.model,
            tokenizer=trainer_core.tokenizer,
        )
    else:
        ui.print_warning("Model or tokenizer not found — chat interface unavailable.")


if __name__ == "__main__":
    main()
