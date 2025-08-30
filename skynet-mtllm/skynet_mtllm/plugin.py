import os
import json
from difflib import SequenceMatcher
from typing import Callable
from jaclang.runtimelib.machine import hookimpl
from mtllm.llm import Model

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from datasets import Dataset

class SkynetMtllmMachine:
    """
    Skynet MTLLM plugin: API-first, dynamic SLM selection, incremental fine-tuning, GPU-aware.
    """

    _cache = {}
    _dataset_file = os.getenv("SKYNET_DATASET", "./.skynet/dataset.jsonl")
    _slm_model_name = os.getenv("SKYNET_SLM_BASE", "gpt2")  # User can change
    _slm_model = None
    _slm_tokenizer = None
    _sim_threshold = float(os.getenv("SKYNET_SIM_THRESHOLD", 0.72))
    _device = 0 if torch.cuda.is_available() else -1

    @staticmethod
    def _init_slm():
        if SkynetMtllmMachine._slm_model is None:
            print(f"[skynet-mtllm] Initializing local SLM ({SkynetMtllmMachine._slm_model_name})")
            SkynetMtllmMachine._slm_tokenizer = AutoTokenizer.from_pretrained(SkynetMtllmMachine._slm_model_name)
            SkynetMtllmMachine._slm_model = AutoModelForCausalLM.from_pretrained(SkynetMtllmMachine._slm_model_name)
            if SkynetMtllmMachine._slm_tokenizer.pad_token is None:
                SkynetMtllmMachine._slm_tokenizer.pad_token = SkynetMtllmMachine._slm_tokenizer.eos_token

    @staticmethod
    def _similar(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _load_dataset():
        if not os.path.exists(SkynetMtllmMachine._dataset_file):
            return Dataset.from_dict({"text": []})
        with open(SkynetMtllmMachine._dataset_file, "r", encoding="utf-8") as f:
            data = [json.loads(line)["prompt"] + " " + json.loads(line)["response"] for line in f]
        return Dataset.from_dict({"text": data})

    @staticmethod
    def _train_slm(prompt: str, response: str):
        """Append prompt-response pair to dataset and fine-tune SLM incrementally."""
        os.makedirs(os.path.dirname(SkynetMtllmMachine._dataset_file), exist_ok=True)
        with open(SkynetMtllmMachine._dataset_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")

        # Fine-tune the local model every N new examples
        dataset = SkynetMtllmMachine._load_dataset()
        if len(dataset) < 5:  # fine-tune only if we have enough examples
            return

        data_collator = DataCollatorForLanguageModeling(tokenizer=SkynetMtllmMachine._slm_tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir="./.skynet/slm_model",
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )
        trainer = Trainer(
            model=SkynetMtllmMachine._slm_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=SkynetMtllmMachine._slm_tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

    @staticmethod
    @hookimpl
    def call_llm(model: Model, caller: Callable, args: dict):
        prompt = list(args.values())[0]
        print(f"[skynet-mtllm] Intercepted call to '{caller.__name__}' with prompt: {prompt}")

        # Check cache
        if prompt in SkynetMtllmMachine._cache:
            print(f"[skynet-mtllm] Using cached response")
            return SkynetMtllmMachine._cache[prompt]

        # Call API first
        api_result = model.invoke(caller, args)
        print(f"[skynet-mtllm] Result fetched from API")

        # Init local SLM
        SkynetMtllmMachine._init_slm()

        # Train SLM incrementally
        SkynetMtllmMachine._train_slm(prompt, api_result)

        # Generate local SLM output for validation
        slm_pipeline = pipeline(
            "text-generation",
            model=SkynetMtllmMachine._slm_model,
            tokenizer=SkynetMtllmMachine._slm_tokenizer,
            device=SkynetMtllmMachine._device
        )
        slm_output = slm_pipeline(prompt, max_length=100, do_sample=False)[0]["generated_text"]

        similarity = SkynetMtllmMachine._similar(api_result, slm_output)
        print(f"[skynet-mtllm] Local SLM similarity with API: {similarity:.2f}")

        # Choose output
        result = slm_output if similarity >= SkynetMtllmMachine._sim_threshold else api_result

        # Cache and return
        SkynetMtllmMachine._cache[prompt] = result
        return result
