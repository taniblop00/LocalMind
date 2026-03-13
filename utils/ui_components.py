from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from collections import deque

console = Console()

class CyberpunkUI:
    def __init__(self):
        self.loss_history = deque(maxlen=30)
        self.thought_stream = deque(maxlen=8)
        self.console = console

    def build_header(self):
        ascii_art = """[bold cyan]
██╗      ██████╗  ██████╗ █████╗ ██╗     ███╗   ███╗██╗███╗   ██╗██████╗ 
██║     ██╔═══██╗██╔════╝██╔══██╗██║     ████╗ ████║██║████╗  ██║██╔══██╗
██║     ██║   ██║██║     ███████║██║     ██╔████╔██║██║██╔██╗ ██║██║  ██║
██║     ██║   ██║██║     ██╔══██║██║     ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
███████╗╚██████╔╝╚██████╗██║  ██║███████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ [/bold cyan]

[black on cyan] 🧪 NEXUS LIVE LABORATORY - ZERO CONFIG FINE-TUNING ENGINE [/black on cyan]
"""
        return Panel(
            Align.center(Text.from_markup(ascii_art)), 
            border_style="cyan",
            padding=(1, 1),
            subtitle="[italic]Accessible & Fully Autonomous AI Research Station[/italic]",
            subtitle_align="right"
        )

    def build_telemetry_table(self, hw_data, phase_name="Training"):
        table = Table(box=None, expand=True, show_edge=False, padding=(0, 2))
        
        table.add_column("Sensor", justify="left", style="bold white", no_wrap=True)
        table.add_column("Reading", justify="right", style="cyan")
        table.add_column("Status", justify="left")

        # GPU Metrics
        if hw_data.get("has_gpu"):
            load = hw_data.get('gpu_load', 0)
            load_status = "[green]Optimal[/]" if load < 90 else "[yellow]Heavy[/]"
            table.add_row("GPU Utilization", f"{load:.1f}%", load_status)
            
            vram_used = hw_data.get('vram_used_gb', 0)
            vram_total = hw_data.get('vram_total_gb', 0)
            vram_pct = (vram_used / vram_total * 100) if vram_total else 0
            v_status = "[green]Nominal[/]" if vram_pct < 85 else "[red]Near Limit[/]"
            table.add_row("VRAM Allocation", f"{vram_used:.1f} / {vram_total:.1f} GB", v_status)
            
            temp = hw_data.get('temp_c', 0)
            temp_status = "[green]Cool[/]" if temp < 70 else ("[yellow]Warm[/]" if temp < 82 else "[red blink]CRITICAL[/]")
            temp_color = "red" if temp > 82 else "cyan"
            table.add_row("Core Temperature", f"[{temp_color}]{temp:.1f}°C[/{temp_color}]", temp_status)
        else:
            table.add_row("Hardware Mode", "CPU Only", "[blue]Active[/]")

        # CPU Metrics
        table.add_row("CPU Utilization", f"{hw_data.get('cpu_load', 0):.1f}%", "[green]Auto-scaled[/]")
        table.add_row("System Memory", f"{hw_data.get('ram_used_gb', 0):.1f} / {hw_data.get('ram_total_gb', 1):.1f} GB", "[white]Managed[/]")
        
        # Spacer
        table.add_row("", "", "")
        
        # Phase Indicator
        table.add_row("Active Protocol", f"[bold cyan]{phase_name}[/]", "[bold green]● Running[/]")
        
        return Panel(
            table, 
            title="[bold white]⚙️ HARDWARE TELEMETRY[/]", 
            border_style="cyan", 
            title_align="left"
        )

    def build_progress_table(self, step, total_steps, current_loss):
        if current_loss is not None:
            self.loss_history.append(current_loss)

        table = Table(box=None, expand=True, show_edge=False, padding=(0, 2))
        table.add_column("Metric", justify="left", style="bold white")
        table.add_column("Value", justify="right", style="cyan")

        progress_pct = (step / total_steps) * 100 if total_steps > 0 else 0
        table.add_row("Step Progress", f"{step} / {total_steps} ({progress_pct:.1f}%)")
        
        if current_loss is not None:
            table.add_row("Current Loss", f"{current_loss:.4f}")
        
        # Visual sparkline for loss trend
        spark_chars = " ▂▃▄▅▆▇█"
        if len(self.loss_history) > 1:
            max_l = max(self.loss_history)
            min_l = min(self.loss_history)
            diff = max_l - min_l if max_l > min_l else 1
            sparkline = ""
            for l in self.loss_history:
                # Invert logic so going down looks like a positive downward trend visual, 
                # or just plot absolute where low is low block, high is high block (standard).
                idx = int(((l - min_l) / diff) * 7)
                # Colorize from green (low) to red (high)
                color = "green" if idx < 3 else "yellow" if idx < 6 else "red"
                sparkline += f"[{color}]{spark_chars[idx]}[/{color}]"
            table.add_row("Loss Trend", sparkline)

        return Panel(
            table, 
            title="[bold white]📈 LEARNING METRICS[/]", 
            border_style="blue",
            title_align="left"
        )

    def build_live_lab_table(self):
        lines = []
        for i, thought in enumerate(self.thought_stream):
            color = "cyan" if i == len(self.thought_stream) - 1 else "grey70"
            lines.append(f"[{color}]» {thought}[/{color}]")
            
        text = "\n".join(lines) if lines else "[grey50]Awaiting data processing instructions...[/grey50]"
        return Panel(
            text, 
            title="[bold white]🧪 ALCHEMY LOG_STREAM[/]", 
            border_style="blue",
            title_align="left"
        )
        
    def build_celebration(self):
        ascii_art = """[bold cyan]

   [bold white]✓ MODEL SYNTHESIS COMPLETE[/bold white]
   ________________________________________________

   The LocalMind neural pathways have been successfully
   calibrated onto the base model using LoRA.
   
   [green]All protocols finalized. Model is ready for deployment.[/green]

[/bold cyan]"""
        return Panel(
            Align.center(Text.from_markup(ascii_art)), 
            title="[bold white]🚀 DEPLOYMENT READY[/bold white]", 
            border_style="green",
            title_align="left"
        )

    def build_footer(self):
        text = Text.from_markup("[grey60]System strictly local. Data remains private. Architecture optimized for accessibility.[/grey60]")
        return Align.center(text)

    def generate_layout(self, hw_data, step=0, total_steps=0, current_loss=None, phase_name="Setup", msg=None):
        if msg:
            self.thought_stream.append(msg)
            
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=10),
            Layout(name="main"),
            Layout(name="footer", size=2)
        )
        layout["main"].split_row(
            Layout(name="telemetry", ratio=1),
            Layout(name="progress", ratio=1)
        )
        
        layout["header"].update(self.build_header())
        layout["telemetry"].update(self.build_telemetry_table(hw_data, phase_name))
        
        if phase_name == "Data Synthesis":
            layout["progress"].update(self.build_live_lab_table())
        elif phase_name == "Training Complete":
            layout["progress"].update(self.build_celebration())
        else:
            layout["progress"].update(self.build_progress_table(step, total_steps, current_loss))
            
        layout["footer"].update(self.build_footer())
        
        return layout

    def print_warning(self, msg):
        self.console.print(f"\n[bold black on yellow] ! WARNING [/bold black on yellow] [yellow]{msg}[/yellow]")

    def print_info(self, msg):
        self.console.print(f"\n[bold black on white] · INFO [/bold black on white] [white]{msg}[/white]")

    def chat_interface(self, model=None, tokenizer=None):
        """Interactive post-training chat interface."""
        self.console.print("\n")
        
        header = "[bold white]🧪 NEXUS INTERACTIVE CONSOLE[/bold white]\n[grey70]Type 'exit' or 'quit' to terminate the session.[/grey70]"
        self.console.print(Panel(header, border_style="cyan", padding=(1, 2)))
        
        if not model or not tokenizer:
            self.print_warning("Model or tokenizer not provided. Inference disabled.")
            return
            
        while True:
            try:
                user_input = self.console.input("\n[bold cyan]YOU >[/bold cyan] ")
                if user_input.lower() in ["exit", "quit", "q"]:
                    self.print_info("Session terminated. Goodbye.")
                    break
                    
                import torch
                
                alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:


### Response:
"""
                formatted_prompt = alpaca_prompt.format(user_input)
                
                inputs = tokenizer(formatted_prompt, return_tensors="pt")
                if hasattr(model, "device"):
                    inputs = inputs.to(model.device)
                
                with self.console.status("[bold cyan]Model is analyzing instructions...[/bold cyan]", spinner="dots", spinner_style="cyan"):
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, 
                            max_new_tokens=256, 
                            use_cache=True, 
                            temperature=0.7, 
                            top_p=0.9
                        )
                        
                input_length = inputs.input_ids.shape[1]
                response_tokens = outputs[0][input_length:]
                response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                
                self.console.print(f"\n[bold green]NEXUS <[/bold green] [white]{response}[/white]\n")
                
            except KeyboardInterrupt:
                self.print_info("Session terminated by user interruption.")
                break
            except Exception as e:
                self.print_warning(f"Inference Error: {str(e)}")
