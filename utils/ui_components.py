import time
from collections import deque
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align

# ─────────────────────────────────────────────────
#  Palette
# ─────────────────────────────────────────────────
C_BORDER    = "#4a9eff"   # Steel-blue border
C_ACCENT    = "#00d4aa"   # Teal accent
C_DIM       = "#5a6a7a"   # Muted text
C_WARN      = "#f5a623"   # Amber
C_OK        = "#3de8a0"   # Mint green
C_ERR       = "#ff5f5f"   # Soft red
C_TITLE     = "#e8f4fd"   # Near-white
C_SPARK_LO  = "#3de8a0"
C_SPARK_MID = "#f5a623"
C_SPARK_HI  = "#ff5f5f"

# ─────────────────────────────────────────────────
#  Singleton console with UTF-8
# ─────────────────────────────────────────────────
console = Console(highlight=False)


def _bar(filled: int, total: int = 30, *, fill="█", empty="░",
         color=C_ACCENT, bg=C_DIM) -> str:
    """Render a single-line progress bar string."""
    n = max(0, min(filled, total))
    return f"[{color}]{'█' * n}[/{color}][{bg}]{'░' * (total - n)}[/{bg}]"


def _eta_str(elapsed: float, done: int, total: int) -> str:
    """Return a human-friendly ETA string."""
    if done <= 0 or total <= 0:
        return "calculating…"
    rate = done / elapsed if elapsed > 0 else 0
    if rate <= 0:
        return "calculating…"
    remaining_secs = (total - done) / rate
    if remaining_secs >= 3600:
        h = int(remaining_secs // 3600)
        m = int((remaining_secs % 3600) // 60)
        return f"~{h}h {m}m remaining"
    elif remaining_secs >= 60:
        m = int(remaining_secs // 60)
        s = int(remaining_secs % 60)
        return f"~{m}m {s:02d}s remaining"
    else:
        return f"~{int(remaining_secs)}s remaining"


# ─────────────────────────────────────────────────
#  UI class
# ─────────────────────────────────────────────────
class CyberpunkUI:
    """
    Lab-grade, flicker-free terminal interface.
    All state is stored on the instance; each Live context
    calls generate_layout() which is idempotent and pure.
    """

    def __init__(self):
        self.console = console

        # ── shared state ──────────────────────────
        self.phase_name   = "Initializing"
        self.phase_start  = time.time()

        # synthesis
        self.synth_log     = deque(maxlen=12)
        self.synth_step    = 0
        self.synth_total   = 0
        self.synth_start   = None

        # training
        self.loss_history  = deque(maxlen=40)
        self.train_step    = 0
        self.train_total   = 0
        self.train_start   = None
        self.current_loss  = None

    # ────────────────────────────────────────────
    #  Public mutators  (called from main.py)
    # ────────────────────────────────────────────
    def push_synth_msg(self, step: int, total: int, msg: str):
        if self.synth_start is None:
            self.synth_start = time.time()
        self.synth_step  = step
        self.synth_total = total
        ts = time.strftime("%H:%M:%S")
        self.synth_log.append((ts, msg))

    def push_train_update(self, step: int, total: int, loss: Optional[float]):
        if self.train_start is None:
            self.train_start = time.time()
        self.train_step  = step
        self.train_total = total
        self.current_loss = loss
        if loss is not None:
            self.loss_history.append(loss)

    def set_phase(self, name: str):
        self.phase_name  = name
        self.phase_start = time.time()

    # ────────────────────────────────────────────
    #  Panel builders  (pure / side-effect free)
    # ────────────────────────────────────────────
    def _build_header(self) -> Panel:
        logo = (
            f"[bold {C_ACCENT}]"
            "  ██╗      ██████╗  ██████╗ █████╗ ██╗     ███╗   ███╗██╗███╗   ██╗██████╗ \n"
            "  ██║     ██╔═══██╗██╔════╝██╔══██╗██║     ████╗ ████║██║████╗  ██║██╔══██╗\n"
            "  ██║     ██║   ██║██║     ███████║██║     ██╔████╔██║██║██╔██╗ ██║██║  ██║\n"
            "  ██║     ██║   ██║██║     ██╔══██║██║     ██║╚██╔╝██║██║██║╚██╗██║██║  ██║\n"
            f"  ███████╗╚██████╔╝╚██████╗██║  ██║███████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝[/bold {C_ACCENT}]\n"
            f"  [dim {C_DIM}]╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝[/dim {C_DIM}]"
        )
        subtitle = (
            f"[{C_DIM}]  Zero-Config Fine-Tuning Engine  ·  "
            f"[bold {C_ACCENT}]{self.phase_name}[/bold {C_ACCENT}]  ·  "
            f"Phase Runtime: {int(time.time() - self.phase_start)}s[/{C_DIM}]"
        )
        content = Text.from_markup(logo + "\n\n" + subtitle)
        return Panel(
            Align.center(content),
            border_style=C_BORDER,
            padding=(0, 2),
        )

    def _build_telemetry(self, hw_data: dict) -> Panel:
        table = Table(box=None, expand=True, show_edge=False,
                      show_header=False, padding=(0, 2))
        table.add_column("Label",   style=f"bold {C_TITLE}", no_wrap=True)
        table.add_column("Value",   justify="right", style=C_ACCENT)
        table.add_column("Status",  justify="left")

        # ── GPU ──────────────────────────────────
        if hw_data.get("has_gpu"):
            load = hw_data.get("gpu_load", 0)
            load_bar = _bar(int(load * 20 / 100), 20)
            table.add_row(
                "  GPU Load",
                f"{load:.1f}%  {load_bar}",
                f"[{C_OK}]Optimal[/{C_OK}]" if load < 90 else f"[{C_WARN}]Heavy[/{C_WARN}]"
            )

            vu = hw_data.get("vram_used_gb", 0)
            vt = hw_data.get("vram_total_gb", 1)
            vr = vu / vt if vt else 0
            v_bar = _bar(int(vr * 20), 20, color=C_OK if vr < 0.85 else C_ERR)
            table.add_row(
                "  VRAM",
                f"{vu:.1f} / {vt:.1f} GB  {v_bar}",
                f"[{C_OK}]Nominal[/{C_OK}]" if vr < 0.85 else f"[{C_ERR}]Near Limit[/{C_ERR}]"
            )

            temp = hw_data.get("temp_c", 0)
            tc = C_OK if temp < 70 else (C_WARN if temp < 82 else C_ERR)
            table.add_row(
                "  Temperature",
                f"[{tc}]{temp:.1f}°C[/{tc}]",
                f"[{C_OK}]Cool[/{C_OK}]" if temp < 70 else (
                    f"[{C_WARN}]Warm[/{C_WARN}]" if temp < 82 else f"[{C_ERR}]CRITICAL[/{C_ERR}]"
                )
            )
        else:
            table.add_row("  Hardware", "CPU Only", f"[{C_ACCENT}]Active[/{C_ACCENT}]")

        # ── CPU ──────────────────────────────────
        cl = hw_data.get("cpu_load", 0)
        cpu_bar = _bar(int(cl * 20 / 100), 20, color=C_ACCENT)
        table.add_row("  CPU", f"{cl:.1f}%  {cpu_bar}", f"[{C_DIM}]Auto-scaled[/{C_DIM}]")

        ru = hw_data.get("ram_used_gb", 0)
        rt = hw_data.get("ram_total_gb", 1)
        rr = ru / rt if rt else 0
        ram_bar = _bar(int(rr * 20), 20, color=C_OK if rr < 0.80 else C_WARN)
        table.add_row(
            "  System RAM",
            f"{ru:.1f} / {rt:.1f} GB  {ram_bar}",
            f"[{C_DIM}]Managed[/{C_DIM}]"
        )

        return Panel(
            table,
            title=f"[bold {C_TITLE}]  Hardware Telemetry[/bold {C_TITLE}]",
            border_style=C_BORDER,
            title_align="left",
            padding=(1, 1),
        )

    def _build_synth_panel(self) -> Panel:
        """Live scrolling log for the Data Synthesis phase."""
        lines = []
        stream = list(self.synth_log)
        for i, (ts, msg) in enumerate(stream):
            is_latest = (i == len(stream) - 1)
            prefix = f"[bold {C_ACCENT}]▶[/bold {C_ACCENT}]" if is_latest else f"[{C_DIM}]·[/{C_DIM}]"
            ts_str = f"[{C_DIM}]{ts}[/{C_DIM}]"
            body   = f"[bold {C_TITLE}]{msg}[/bold {C_TITLE}]" if is_latest else f"[{C_DIM}]{msg}[/{C_DIM}]"
            lines.append(f"  {prefix}  {ts_str}  {body}")

        if not lines:
            lines.append(f"  [{C_DIM}]Waiting for chunks…[/{C_DIM}]")

        # ── Progress bar ─────────────────────────
        if self.synth_total > 0:
            pct    = self.synth_step / self.synth_total
            filled = int(pct * 34)
            bar    = _bar(filled, 34)
            elapsed = (time.time() - self.synth_start) if self.synth_start else 0
            eta    = _eta_str(elapsed, self.synth_step, self.synth_total)
            chunk_info = (
                f"\n\n  [{C_DIM}]Chunks[/{C_DIM}]  {bar}  "
                f"[bold {C_ACCENT}]{self.synth_step}[/bold {C_ACCENT}]"
                f"[{C_DIM}]/{self.synth_total}[/{C_DIM}]"
                f"  [{C_DIM}]{pct*100:.0f}%  ·  {eta}[/{C_DIM}]"
            )
            lines.append(chunk_info)

        return Panel(
            "\n".join(lines),
            title=f"[bold {C_TITLE}]  Data Synthesis  ·  Live Log[/bold {C_TITLE}]",
            border_style=C_ACCENT,
            title_align="left",
            padding=(1, 1),
        )

    def _build_train_panel(self) -> Panel:
        """Training progress with sparkline and ETA."""
        table = Table(box=None, expand=True, show_edge=False,
                      show_header=False, padding=(0, 2))
        table.add_column("Label", style=f"bold {C_TITLE}", no_wrap=True)
        table.add_column("Value", justify="left", style=C_ACCENT)

        # Step progress
        step  = self.train_step
        total = self.train_total
        pct   = (step / total * 100) if total > 0 else 0
        filled = int(pct * 34 / 100)
        bar    = _bar(filled, 34)
        elapsed = (time.time() - self.train_start) if self.train_start else 0
        eta    = _eta_str(elapsed, step, total)

        table.add_row(
            "  Step",
            (
                f"[bold {C_ACCENT}]{step}[/bold {C_ACCENT}]"
                f"[{C_DIM}] / {total}[/{C_DIM}]"
                f"  [{C_DIM}]({pct:.1f}%)[/{C_DIM}]"
            )
        )
        table.add_row("  Progress", f"{bar}")
        table.add_row("  ETA", f"[{C_DIM}]{eta}[/{C_DIM}]")

        if self.current_loss is not None:
            loss_color = C_OK if self.current_loss < 1.5 else (
                C_WARN if self.current_loss < 3.0 else C_ERR
            )
            table.add_row(
                "  Current Loss",
                f"[bold {loss_color}]{self.current_loss:.4f}[/bold {loss_color}]"
            )

        # Sparkline
        spark_chars = " ▁▂▃▄▅▆▇█"
        hist = list(self.loss_history)
        if len(hist) > 1:
            mn, mx = min(hist), max(hist)
            diff = mx - mn if mx > mn else 1
            spark = ""
            for l in hist[-36:]:
                idx = int(((l - mn) / diff) * 8)
                idx = max(0, min(idx, 8))
                col = C_SPARK_LO if idx < 3 else (
                    C_SPARK_MID if idx < 6 else C_SPARK_HI
                )
                spark += f"[{col}]{spark_chars[idx]}[/{col}]"
            table.add_row("  Loss Trend", spark)

        return Panel(
            table,
            title=f"[bold {C_TITLE}]  Model Alignment  ·  Training Metrics[/bold {C_TITLE}]",
            border_style=C_BORDER,
            title_align="left",
            padding=(1, 1),
        )

    def _build_idle_panel(self, phase_name: str) -> Panel:
        """Generic idle / init panel for phases without specific progress."""
        elapsed = int(time.time() - self.phase_start)
        dot = "●" if elapsed % 2 == 0 else "○"
        body = (
            f"\n  [{C_DIM}]{dot}  Phase[/{C_DIM}]  "
            f"[bold {C_ACCENT}]{phase_name}[/bold {C_ACCENT}]  "
            f"[{C_DIM}]is active  ·  {elapsed}s elapsed[/{C_DIM}]\n"
        )
        return Panel(
            body,
            title=f"[bold {C_TITLE}]  System Status[/bold {C_TITLE}]",
            border_style=C_DIM,
            title_align="left",
            padding=(1, 1),
        )

    def _build_done_panel(self) -> Panel:
        body = (
            f"\n"
            f"  [bold {C_OK}]✓  Model synthesis complete.[/bold {C_OK}]\n\n"
            f"  [{C_DIM}]The LocalMind neural pathways have been successfully\n"
            f"  calibrated onto the base model using LoRA.\n\n"
            f"  All protocols finalized — model is ready for deployment.[/{C_DIM}]\n"
        )
        return Panel(
            body,
            title=f"[bold {C_OK}]  Deployment Ready[/bold {C_OK}]",
            border_style=C_OK,
            title_align="left",
            padding=(1, 1),
        )

    def _build_footer(self) -> Text:
        return Text.from_markup(
            f"  [{C_DIM}]System strictly local  ·  Data remains private  ·  "
            f"Architecture optimized for accessibility  ·  taniblop00[/{C_DIM}]"
        )

    # ────────────────────────────────────────────
    #  Master layout composer
    # ────────────────────────────────────────────
    def generate_layout(
        self,
        hw_data: dict,
        *,
        step: int = 0,
        total_steps: int = 0,
        current_loss=None,
        phase_name: Optional[str] = None,
        msg: Optional[str] = None,
        synth_step: int = 0,
        synth_total: int = 0,
    ) -> Layout:
        """
        Build and return the complete screen layout.
        Pure function — no side-effects apart from calling push_* helpers.
        """
        # ── propagate incoming state ──────────────
        if phase_name and phase_name != self.phase_name:
            self.set_phase(phase_name)
        elif phase_name is None:
            phase_name = self.phase_name

        if msg:
            self.push_synth_msg(synth_step, synth_total, msg)
        elif synth_total > 0:
            self.synth_step  = synth_step
            self.synth_total = synth_total

        if total_steps > 0 or step > 0:
            self.push_train_update(step, total_steps, current_loss)

        # ── assemble layout ───────────────────────
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=10),
            Layout(name="main"),
            Layout(name="footer", size=1),
        )
        layout["main"].split_row(
            Layout(name="left",  ratio=5),
            Layout(name="right", ratio=5),
        )

        layout["header"].update(self._build_header())
        layout["left"].update(self._build_telemetry(hw_data))
        layout["footer"].update(self._build_footer())

        pn = self.phase_name
        if pn == "Data Synthesis":
            layout["right"].update(self._build_synth_panel())
        elif pn == "Model Alignment":
            layout["right"].update(self._build_train_panel())
        elif pn == "Training Complete":
            layout["right"].update(self._build_done_panel())
        else:
            layout["right"].update(self._build_idle_panel(pn))

        return layout

    # ────────────────────────────────────────────
    #  Simple print helpers (outside of Live ctx)
    # ────────────────────────────────────────────
    def print_warning(self, msg: str):
        self.console.print(
            f"\n  [{C_WARN}]⚠  {msg}[/{C_WARN}]"
        )

    def print_info(self, msg: str):
        self.console.print(
            f"\n  [{C_ACCENT}]→  {msg}[/{C_ACCENT}]"
        )

    # ────────────────────────────────────────────
    #  Post-training chat interface
    # ────────────────────────────────────────────
    def chat_interface(self, model=None, tokenizer=None):
        """Interactive chat REPL after training is complete."""
        self.console.print("\n")
        self.console.print(Panel(
            f"[bold {C_TITLE}]LocalMind Interactive Console[/bold {C_TITLE}]\n"
            f"[{C_DIM}]Type 'exit' or 'quit' to end the session.[/{C_DIM}]",
            border_style=C_BORDER,
            padding=(1, 2),
        ))

        if not model or not tokenizer:
            self.print_warning("Model or tokenizer not available — inference disabled.")
            return

        alpaca_prompt = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            "### Instruction:\n{}\n\n### Input:\n\n\n### Response:\n"
        )

        while True:
            try:
                user_input = self.console.input(
                    f"\n  [bold {C_ACCENT}]YOU ›[/bold {C_ACCENT}]  "
                )
                if user_input.strip().lower() in {"exit", "quit", "q"}:
                    self.print_info("Session terminated. Goodbye.")
                    break

                import torch
                prompt = alpaca_prompt.format(user_input)
                inputs = tokenizer(prompt, return_tensors="pt")
                if hasattr(model, "device"):
                    inputs = inputs.to(model.device)

                with self.console.status(
                    f"  [{C_DIM}]Analyzing…[/{C_DIM}]",
                    spinner="dots",
                    spinner_style=C_ACCENT,
                ):
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            use_cache=True,
                            temperature=0.7,
                            top_p=0.9,
                        )

                resp = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                self.console.print(
                    f"\n  [bold {C_OK}]NEXUS ‹[/bold {C_OK}]  [{C_TITLE}]{resp}[/{C_TITLE}]\n"
                )

            except KeyboardInterrupt:
                self.print_info("Session interrupted.")
                break
            except Exception as e:
                self.print_warning(f"Inference error: {e}")
