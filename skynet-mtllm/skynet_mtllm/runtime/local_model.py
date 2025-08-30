import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalModel:
    def __init__(self, base_model: str, workdir, device: str):
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
            self.ready = True
        except Exception:
            self.ready = False

    def generate(self, prompt: str) -> str:
        if not self.ready:
            raise RuntimeError("Local model not ready")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
