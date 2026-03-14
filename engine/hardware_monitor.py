import os
import psutil
try:
    import pynvml
    import torch
    # BUG FIX: must also confirm CUDA is actually available, not just that torch is installed
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

class HardwareManager:
    """
    Automatic Profiling and Hardware-Driven Decision Engine.
    Detects VRAM, RAM, CPU Topologies, and configures the host optimally.
    """
    def __init__(self):
        self.has_gpu = HAS_GPU
        self.gpu_name = "N/A"
        self.total_vram_gb = 0.0
        
        self.total_ram_gb = 0.0
        self.cpu_physical_cores = psutil.cpu_count(logical=False) or 1
        self.cpu_logical_cores = psutil.cpu_count(logical=True) or 1
        
        self.compute_capability = "N/A"
        self.thermal_threshold_c = 82
        
        self._initialize_hardware()
        self.config = self.get_optimal_config()

    def _initialize_hardware(self):
        # RAM
        self.total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # GPU
        if self.has_gpu and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(self.gpu_name, bytes):
                    self.gpu_name = self.gpu_name.decode("utf-8")
                    
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.total_vram_gb = info.total / (1024 ** 3)
                
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                self.compute_capability = f"{major}.{minor}"
            except Exception as e:
                self._handle_gpu_error(e, "Initialization")
        else:
            self.has_gpu = False

    def _handle_gpu_error(self, e, context: str):
        """Plain English/Hebrew error handling for GPU calls."""
        msg = f"[ERROR / שגיאה] GPU {context} Failed."
        msg_he = "אירעה שגיאה בתקשורת עם כרטיס המסך (GPU)."
        msg_en = f"An error occurred while communicating with the GPU. Details: {str(e)}"
        print(f"\033[91m{msg}\n{msg_he}\n{msg_en}\033[0m")
        self.has_gpu = False

    def get_telemetry(self):
        """Returns concurrent GPU and CPU Load."""
        telemetry = {
            "has_gpu": self.has_gpu,
            "gpu_load": 0.0,
            "vram_used_gb": 0.0,
            "vram_total_gb": self.total_vram_gb,
            "temp_c": 0.0,
            "cpu_load": psutil.cpu_percent(),
            "ram_used_gb": psutil.virtual_memory().used / (1024 ** 3),
            "ram_total_gb": self.total_ram_gb
        }

        if self.has_gpu:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                telemetry["gpu_load"] = util.gpu
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                telemetry["vram_used_gb"] = mem_info.used / (1024 ** 3)
                
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                telemetry["temp_c"] = temp
            except Exception as e:
                self._handle_gpu_error(e, "Telemetry")
                
        return telemetry

    def check_thermal_protection(self):
        """Checks if GPU is overheating."""
        if not self.has_gpu:
            return False, 0.0
            
        telemetry = self.get_telemetry()
        temp = telemetry["temp_c"]
        is_overheating = temp > self.thermal_threshold_c
        return is_overheating, temp

    def get_optimal_config(self):
        """Returns the model mapping based on CPU Topologies / VRAM."""
        config = {
            "mode": "cpu",
            "model_name": "N/A",
            "quantization": False,
            "lora_rank": 8,
            "gradient_checkpointing": False,
            "cpu_threads": self.cpu_physical_cores,
            "dataloader_workers": max(1, self.cpu_physical_cores // 2)
        }

        if not self.has_gpu:
            config["mode"] = "cpu"
            # Switch to a more capable, but still small model for CPU fallback if possible.
            config["model_name"] = "HuggingFaceTB/SmolLM-135M-Instruct" 
            config["quantization"] = False  # bitsandbytes handles 8bit on cpu, handled in trainer
            config["lora_rank"] = 4
            return config

        config["mode"] = "gpu"
        vram = self.total_vram_gb

        if vram < 6.0:
            config["model_name"] = "unsloth/tinyllama-bnb-4bit"
            config["quantization"] = True
            config["lora_rank"] = 8
            config["gradient_checkpointing"] = "unsloth"
        elif 6.0 <= vram < 14.0:
            config["model_name"] = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
            config["quantization"] = True
            config["lora_rank"] = 16
            config["gradient_checkpointing"] = "unsloth"
        else: # >= 14GB 
            config["model_name"] = "unsloth/Llama-3.1-8B-bnb-4bit"
            config["quantization"] = True
            config["lora_rank"] = 32
            config["gradient_checkpointing"] = "unsloth"

        if vram < 8.0:
            config["gradient_checkpointing"] = "unsloth"

        return config
