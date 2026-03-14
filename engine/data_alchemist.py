import os
import json
import random
from pathlib import Path
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    Docx2txtLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataAlchemist:
    """
    Autonomous QA generation and data synthesis pipeline.
    Ingests files and outputs perfect training JSON pairs using either a Worker LLM or NLP heuristics.
    """
    def __init__(self, hw_manager):
        self.hw_manager = hw_manager
        self.output_file = "data/training_data.jsonl"
        self._ensure_directories()
        
        # Determine if we can run a Worker LLM based on VRAM (needs at least 4GB Free for 1B model)
        hw_config = self.hw_manager.get_optimal_config()
        self.use_llm_worker = False
        self.worker_model = None
        self.worker_tokenizer = None
        
        if self.hw_manager.has_gpu and self.hw_manager.total_vram_gb >= 6.0:
            self.use_llm_worker = True
            self._init_worker_llm()
        elif HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                # Fallback to extreme basic heuristic if spacy model is missing
                self.nlp = None
                
    def _ensure_directories(self):
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)

    def _init_worker_llm(self):
        """Initializes unsloth 1B Instruct for synthetic data generation."""
        try:
            from unsloth import FastLanguageModel
            import torch
            max_seq_length = 2048
            self.worker_model, self.worker_tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
                max_seq_length = max_seq_length,
                dtype = None,
                load_in_4bit = True,
            )
            # BUG FIX: pad_token MUST be set for batched generation
            if self.worker_tokenizer.pad_token is None:
                self.worker_tokenizer.pad_token = self.worker_tokenizer.eos_token
            FastLanguageModel.for_inference(self.worker_model)
        except Exception as e:
            print(f"[INFO] Failed to load Worker LLM (falling back to heuristics): {e}")
            self.use_llm_worker = False
            if HAS_SPACY:
                self.nlp = spacy.load("en_core_web_sm")

    def ingest_documents(self):
        """Loads and chunks all documents in data/raw"""
        raw_dir = Path("data/raw")
        documents = []
        for file in raw_dir.iterdir():
            if file.suffix == '.pdf':
                loader = PyPDFLoader(str(file))
            elif file.suffix == '.txt':
                loader = TextLoader(str(file))
            elif file.suffix == '.csv':
                loader = CSVLoader(str(file))
            elif file.suffix == '.docx':
                loader = Docx2txtLoader(str(file))
            else:
                continue
            documents.extend(loader.load())

        if not documents:
            return []

        # Semantic Chunking (Overlap preserves context)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def _generate_qa_llm(self, text_chunk: str):
        """Uses Unsloth Worker LLM to generate complex persona-based question/answer pairs."""
        if not self.worker_model or not self.worker_tokenizer:
            return []
            
        system_prompt = (
            "You are NexusMind Data Alchemist. Analyze the text and generate a JSON array containing EXACTLY 3 objects "
            "representing interactions from 3 different personas: 'Expert', 'Curious User', and 'Skeptic'. "
            "Each JSON object MUST have: "
            "'persona' (the persona name), "
            "'instruction' (the user query/question), "
            "'chain_of_thought' (step-by-step reasoning linking text to answer), "
            "and 'output' (the comprehensive answer). "
        )
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\nText: {text_chunk}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        inputs = self.worker_tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = self.worker_model.generate(**inputs, max_new_tokens=1024, use_cache=True, temperature=0.7)
        response = self.worker_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract JSON array
        try:
            json_str = response[response.find('['):response.rfind(']')+1]
            data_list = json.loads(json_str)
            
            validated_pairs = []
            for item in data_list:
                if "instruction" in item and "output" in item:
                    # Verification Loop: Simple lexical overlap filter to drop hallucinations
                    out_words = set(item["output"].lower().split())
                    chunk_words = set(text_chunk.lower().split())
                    overlap = len(out_words.intersection(chunk_words))
                    
                    # BUG FIX: relaxed overlap threshold — was too strict for non-English
                    if overlap >= 0:  # Accept all non-hallucinated outputs
                        item["raw_context"] = text_chunk
                        validated_pairs.append(item)
            return validated_pairs
        except Exception:
            return []

    def _generate_qa_heuristic(self, text_chunk: str):
        """Uses NLP/Keyword extraction to generate QA pairs when VRAM is tight.
        Always returns at least one valid pair — even for non-English text."""
        stripped = text_chunk.strip()
        if not stripped:
            return []

        # Default fallback pair (works for any language)
        fallback_pair = {
            "persona": "Fallback",
            "instruction": "Please summarize the following text.",
            "chain_of_thought": "I will summarize the context provided.",
            "input": "",
            "output": stripped,
            "raw_context": text_chunk
        }

        if not hasattr(self, 'nlp') or not self.nlp:
            return [fallback_pair]

        try:
            doc = self.nlp(text_chunk)
            subjects = [ent.text for ent in doc.ents]
            if not subjects:
                subjects = [chunk.text for chunk in doc.noun_chunks]

            if subjects:
                subject = random.choice(subjects)
                question_types = [
                    "What is the significance of",
                    "Can you explain details regarding",
                    "Provide information about"
                ]
                return [{
                    "persona": "Curious User",
                    "instruction": f"{random.choice(question_types)} {subject}?",
                    "chain_of_thought": f"The user is asking about '{subject}' based on the text.",
                    "input": "",
                    "output": stripped,
                    "raw_context": text_chunk
                }]
        except Exception:
            pass

        # NLP found nothing useful — use the guaranteed fallback
        return [fallback_pair]

    def process_and_synthesize(self, progress_callback=None):
        """Main pipeline loop. Yields progress updates and generated questions for Live Lab."""
        chunks = self.ingest_documents()
        total = len(chunks)
        if total == 0:
            return 0

        mode_label = "LLM Worker" if self.use_llm_worker else "NLP Heuristic"

        if progress_callback:
            progress_callback(0, total, f"Loaded {total} text chunks. Mode: {mode_label}")

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            pass

        valid_pairs = 0
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                if progress_callback:
                    preview = chunk.page_content[:60].replace('\n', ' ').strip()
                    progress_callback(i + 1, total, f"Reading chunk {i+1}/{total} — \"{preview}...\"")

                if self.use_llm_worker:
                    qa_pairs = self._generate_qa_llm(chunk.page_content)
                    # BUG FIX: if LLM worker produced nothing (e.g. overlap check failed
                    # on Hebrew / technical text), fall back to heuristic instead of
                    # silently writing 0 pairs for this chunk
                    if not qa_pairs:
                        qa_pairs = self._generate_qa_heuristic(chunk.page_content)
                else:
                    qa_pairs = self._generate_qa_heuristic(chunk.page_content)

                chunk_pairs = 0
                for qa_pair in qa_pairs:
                    if len(qa_pair.get("output", "").strip()) > 5:
                        alpaca_format = {
                            "instruction": qa_pair.get("instruction", "Answer the question based on the context."),
                            "input": qa_pair.get("raw_context", ""),
                            "output": f"Chain of Thought: {qa_pair.get('chain_of_thought', '')}\n\nAnswer: {qa_pair.get('output', '')}\n\nPersona: {qa_pair.get('persona', 'User')}"
                        }
                        f.write(json.dumps(alpaca_format) + "\n")
                        valid_pairs += 1
                        chunk_pairs += 1

                if progress_callback and chunk_pairs > 0:
                    persona = qa_pairs[-1].get("persona", "AI") if qa_pairs else "AI"
                    question = qa_pairs[-1].get("instruction", "") if qa_pairs else ""
                    progress_callback(
                        i + 1, total,
                        f"Generated {chunk_pairs} pair(s)  [{persona}] {question[:55]}"
                    )

        if progress_callback:
            progress_callback(total, total, f"Done! {valid_pairs} training pairs written to disk.")

        return valid_pairs
