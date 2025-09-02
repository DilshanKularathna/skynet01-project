import os
import json
from difflib import SequenceMatcher
from typing import Callable
from jaclang.runtimelib.machine import hookimpl
from mtllm.llm import Model

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


class SkynetMtllmMachine:
    """
    Skynet MTLLM plugin: API-first, dynamic SLM selection, incremental fine-tuning with LoRA, GPU-aware.
    """

    _cache = {}
    _dataset_file = os.getenv("SKYNET_DATASET", "./.skynet/dataset.jsonl")
    _slm_model_name = os.getenv("SKYNET_SLM_BASE", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # Hugging Face model ID
    _slm_model = None
    _slm_tokenizer = None
    _sim_threshold = float(os.getenv("SKYNET_SIM_THRESHOLD", 0.72))
    _device = 0 if torch.cuda.is_available() else -1
    _use_api = os.getenv("SKYNET_USE_API", "true").lower() == "true"
    _train_local = os.getenv("SKYNET_TRAIN_LOCAL", "true").lower() == "true"

    @staticmethod
    def _init_slm():
        if SkynetMtllmMachine._slm_model is None:
            print(f"[skynet-mtllm] Initializing local SLM ({SkynetMtllmMachine._slm_model_name})")
            SkynetMtllmMachine._slm_tokenizer = AutoTokenizer.from_pretrained(
                SkynetMtllmMachine._slm_model_name
            )
            SkynetMtllmMachine._slm_model = AutoModelForCausalLM.from_pretrained(
                SkynetMtllmMachine._slm_model_name
            )
            if SkynetMtllmMachine._slm_tokenizer.pad_token is None:
                SkynetMtllmMachine._slm_tokenizer.pad_token = SkynetMtllmMachine._slm_tokenizer.eos_token

            # Move model to correct device
            if SkynetMtllmMachine._device >= 0:
                SkynetMtllmMachine._slm_model.to(f"cuda:{SkynetMtllmMachine._device}")
            else:
                SkynetMtllmMachine._slm_model.to("cpu")

    @staticmethod
    def _similar(a: str, b: str) -> float:
        """Calculate similarity between two strings."""
        # Ensure both inputs are strings
        str_a = str(a) if a is not None else ""
        str_b = str(b) if b is not None else ""
        return SequenceMatcher(None, str_a, str_b).ratio()

    @staticmethod
    def _serialize(obj):
        """Safely convert object to JSON-serializable form."""
        if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
            return obj
        # fallback: convert custom objects to dict if possible, else string
        return getattr(obj, "_dict_", str(obj))

    @staticmethod
    def _to_string(obj):
        """Convert any object to a string representation for comparison."""
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, (dict, list)):
            return json.dumps(obj, sort_keys=True)
        else:
            # For custom objects like Level, convert to JSON string
            try:
                return json.dumps(SkynetMtllmMachine._serialize(obj), sort_keys=True)
            except (TypeError, ValueError):
                return str(obj)

    @staticmethod
    def _load_dataset():
        if not os.path.exists(SkynetMtllmMachine._dataset_file):
            return Dataset.from_dict({"text": []})

        data = []
        with open(SkynetMtllmMachine._dataset_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    prompt_text = SkynetMtllmMachine._serialize(entry.get("prompt", ""))
                    response_text = SkynetMtllmMachine._serialize(entry.get("response", ""))
                    data.append(str(prompt_text) + " " + str(response_text))
                except json.JSONDecodeError:
                    continue
        return Dataset.from_dict({"text": data})

    @staticmethod
    def _train_slm(prompt: str, response: str):
        """Append prompt-response pair to dataset and fine-tune SLM incrementally with LoRA."""
        if not SkynetMtllmMachine._train_local:
            return

        try:
            os.makedirs(os.path.dirname(SkynetMtllmMachine._dataset_file), exist_ok=True)
            with open(SkynetMtllmMachine._dataset_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "prompt": SkynetMtllmMachine._serialize(prompt),
                    "response": SkynetMtllmMachine._serialize(response)
                }) + "\n")

            dataset = SkynetMtllmMachine._load_dataset()
            if len(dataset) < 5:
                return

            # Tokenize dataset
            def tokenize_fn(example):
                return SkynetMtllmMachine._slm_tokenizer(
                    example["text"], truncation=True, padding="max_length", max_length=128
                )

            tokenized_dataset = dataset.map(tokenize_fn, batched=True)

            # Apply LoRA
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],  # TinyLLama compatible
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            lora_model = get_peft_model(SkynetMtllmMachine._slm_model, lora_config)

            # Move LoRA model to correct device
            if SkynetMtllmMachine._device >= 0:
                lora_model.to(f"cuda:{SkynetMtllmMachine._device}")
            else:
                lora_model.to("cpu")

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=SkynetMtllmMachine._slm_tokenizer, mlm=False
            )

            training_args = TrainingArguments(
                output_dir=os.path.join(os.path.dirname(SkynetMtllmMachine._dataset_file), "slm_model"),
                per_device_train_batch_size=1,
                num_train_epochs=1,
                logging_steps=10,
                save_strategy="no",
                report_to="none",
            )

            trainer = Trainer(
                model=lora_model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=SkynetMtllmMachine._slm_tokenizer,
                data_collator=data_collator,
            )
            trainer.train()

            # Save LoRA adapters
            lora_model.save_pretrained("./.skynet/slm_model_lora")
            SkynetMtllmMachine._slm_model = lora_model
            
        except Exception as e:
            print(f"[skynet-mtllm] Training error: {e}")

    @staticmethod
    @hookimpl
    def call_llm(model: Model, caller: Callable, args: dict):
        try:
            prompt = list(args.values())[0] if args else ""
            caller_name = getattr(caller, '__name__', str(caller))
            print(f"[skynet-mtllm] Intercepted call to '{caller_name}' with prompt: {prompt}")

            # 🔑 Normalize prompt into a string key
            if isinstance(prompt, (list, dict)):
                prompt_key = json.dumps(SkynetMtllmMachine._serialize(prompt), sort_keys=True)
            else:
                prompt_key = str(prompt)
            # include caller name to avoid collisions
            cache_key = f"{caller_name}:{prompt_key}"

            # Use cache if available
            if cache_key in SkynetMtllmMachine._cache:
                print(f"[skynet-mtllm] Using cached response")
                return SkynetMtllmMachine._cache[cache_key]

            # Call API first if enabled
            api_result = None
            if SkynetMtllmMachine._use_api:
                api_result = model.invoke(caller, args)
                print(f"[skynet-mtllm] Result fetched from API")

            # Initialize local SLM
            SkynetMtllmMachine._init_slm()
            
            # Train SLM with API result if available
            if api_result is not None:
                SkynetMtllmMachine._train_slm(prompt, api_result)

            # Generate local SLM output
            slm_pipeline = pipeline(
                "text-generation",
                model=SkynetMtllmMachine._slm_model,
                tokenizer=SkynetMtllmMachine._slm_tokenizer,
                device=SkynetMtllmMachine._device,
            )
            slm_output = slm_pipeline(str(prompt), max_length=100, do_sample=False)[0]["generated_text"]

            # Convert both results to strings for comparison
            api_result_str = SkynetMtllmMachine._to_string(api_result) if api_result is not None else ""
            slm_output_str = SkynetMtllmMachine._to_string(slm_output)
            
            # Calculate similarity only if we have API result
            if api_result is not None:
                similarity = SkynetMtllmMachine._similar(api_result_str, slm_output_str)
                print(f"[skynet-mtllm] Local SLM similarity with API: {similarity:.2f}")
                
                # Use local model if similarity exceeds threshold, otherwise use API result
                result = slm_output if similarity >= SkynetMtllmMachine._sim_threshold else api_result
            else:
                # If no API result, use SLM output
                result = slm_output
                print(f"[skynet-mtllm] Using local SLM output (no API call)")

            # Cache and return result
            SkynetMtllmMachine._cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"[skynet-mtllm] Error in call_llm: {e}")
            # Fallback to original model call
            return model.invoke(caller, args)