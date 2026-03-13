import os
# Forcefully silence HuggingFace, Datasets, and TensorFlow before they load
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("trl").setLevel(logging.ERROR)

from datasets import load_dataset
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from trl import SFTTrainer
    from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

class TrainerCore:
    """
    Unsloth-powered (or CPU Peft fallback) training logic.
    Handles dynamic hyper-parameters and model merging.
    """
    def __init__(self, hw_manager):
        self.hw_manager = hw_manager
        self.config = hw_manager.get_optimal_config()
        self.max_seq_length = 2048
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def format_prompts(self, examples):
        """Alpaca prompt formatting."""
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + self.tokenizer.eos_token
            texts.append(text)
        return { "text" : texts, }

    def setup_model(self):
        """Loads model relying on dynamic HW selection."""
        model_name = self.config["model_name"]
        use_quantization = self.config["quantization"]
        
        if self.config["mode"] == "gpu" and UNSLOTH_AVAILABLE:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_name,
                max_seq_length = self.max_seq_length,
                dtype = None, # Auto detection
                load_in_4bit = use_quantization,
            )
            
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r = self.config["lora_rank"],
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj",],
                lora_alpha = 16,
                lora_dropout = 0, # Unsloth supports only 0
                bias = "none",
                use_gradient_checkpointing = self.config["gradient_checkpointing"],
                random_state = 3407,
                use_rslora = False,
                loftq_config = None,
            )
        else:
            # CPU Fallback via Huggingface PEFT
            # Provide silent setup
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            try:
                from transformers import BitsAndBytesConfig
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map="cpu", 
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map="cpu", 
                    torch_dtype=torch.float32
                )
            peft_config = LoraConfig(
                r=self.config["lora_rank"],
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, peft_config)

    def _calculate_dynamic_params(self, dataset_len):
        """Calculates batch size and epochs based on hardware and dataset size."""
        # Prevent overfitting on tiny datasets
        epochs = 1
        if dataset_len < 50:
             epochs = 3
        elif dataset_len < 200:
             epochs = 2
             
        # Target 90% VRAM (Heuristic based on rank and sequence length)
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 4
        
        if self.config["mode"] == "gpu":
            if self.hw_manager.total_vram_gb > 16:
                per_device_train_batch_size = 4
                gradient_accumulation_steps = 2
            elif self.hw_manager.total_vram_gb < 8:
                per_device_train_batch_size = 1
                gradient_accumulation_steps = 8
                
        # Handle CPU explicitly
        if self.config["mode"] == "cpu":
             per_device_train_batch_size = 1
             gradient_accumulation_steps = 8
             
        return epochs, per_device_train_batch_size, gradient_accumulation_steps

    def start_training(self, data_path="data/training_data.jsonl", callback=None):
        """Executes training and reports metrics to callback."""
        if not os.path.exists(data_path):
             raise FileNotFoundError("Training data not found.")
             
        from datasets import disable_progress_bar
        disable_progress_bar()
        
        dataset = load_dataset("json", data_files=data_path, split="train")
        dataset = dataset.map(self.format_prompts, batched=True, desc="Mapping prompt")
        
        epochs, batch_size, grad_acc = self._calculate_dynamic_params(len(dataset))

        # We construct a custom callback to bridge HuggingFace and Rich TUI
        from transformers import TrainerCallback
        class TuiBridgeCallback(TrainerCallback):
             def on_log(self, args, state, control, logs=None, **kwargs):
                 if callback and logs and "loss" in logs:
                      # state.global_step, state.max_steps, logs["loss"]
                      callback(state.global_step, state.max_steps, logs["loss"])

        training_args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_acc,
            warmup_steps = 5,
            num_train_epochs = epochs,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported() if UNSLOTH_AVAILABLE and self.config["mode"] == "gpu" else False,
            bf16 = is_bfloat16_supported() if UNSLOTH_AVAILABLE and self.config["mode"] == "gpu" else False,
            logging_steps = 1,
            optim = "adamw_8bit" if self.config["mode"] == "gpu" else "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            dataloader_num_workers = self.config.get("dataloader_workers", 1),
            disable_tqdm=True,              # Hide standard progress bars
            report_to="none",               # Don't log to W&B / stdout
            log_level="error",              # Hide info/warnings from trainer
        )

        import logging
        from datasets import disable_progress_bar
        disable_progress_bar()              # Hide datasets map progress bar
        logging.getLogger("transformers").setLevel(logging.ERROR) # Suppress warnings

        self.trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for Short sequences.
            args = training_args,
            callbacks=[TuiBridgeCallback()]
        )
        
        import contextlib
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                self.trainer.train()

    def save_and_export(self):
        """Saves LoRA weights, attempts GGUF export, and generates standalone inference script."""
        try:
            self.model.save_pretrained("outputs/lora_model")
            self.tokenizer.save_pretrained("outputs/lora_model")
            
            if self.config["mode"] == "gpu" and UNSLOTH_AVAILABLE:
                try:
                    # GGUF Export
                    self.model.save_pretrained_gguf("outputs/model", self.tokenizer, quantization_method = "q4_k_m")
                except Exception as e:
                    if hasattr(self.hw_manager, "print_warning"):
                        self.hw_manager.print_warning(f"GGUF Export failed. C++ tools might be missing: {e}")
                    else:
                        print(f"[WARNING] GGUF Export failed: {e}")
            
            self._generate_inference_script()
        except Exception as e:
            print(f"[ERROR] Export Failed: {e}")

    def _generate_inference_script(self):
        script_content = f"""import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
lora_dir = os.path.join(script_dir, "lora_model")

print("Loading {self.config['model_name']} with NexusMind LoRA...")
tokenizer = AutoTokenizer.from_pretrained(lora_dir)
model = AutoModelForCausalLM.from_pretrained("{self.config['model_name']}", device_map="auto")
model = PeftModel.from_pretrained(model, lora_dir)

print("Model loaded successfully. Enter your prompt:")
while True:
    prompt = input("> ")
    if prompt.lower() in ["exit", "quit", "q"]: break
    
    alpaca_prompt = \"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{{}}

### Input:


### Response:
\"\"\"
    formatted_prompt = alpaca_prompt.format(prompt)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, use_cache=True)
    
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    print(response.strip())
"""
        with open("outputs/inference.py", "w", encoding="utf-8") as f:
            f.write(script_content)
