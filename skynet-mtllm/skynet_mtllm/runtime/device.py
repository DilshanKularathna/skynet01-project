import torch, warnings

def detect_device():
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple GPU
        else:
            return "cpu"
    except Exception:
        warnings.warn("[skynet] torch not installed, defaulting to CPU")
        return "cpu"
