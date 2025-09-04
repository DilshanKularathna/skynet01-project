# Skynet MTLLM Plugin

**Skynet MTLLM** is an extension plugin for [Jaclang MTLLM](https://github.com/jaseci-labs/jaseci) that adds:

- Dynamic model switching between API-based LLMs and local SLMs (Small Language Models)
- GPU support for NVIDIA (CUDA) and AMD (ROCm)
- Dataset logging and automated training of local SLMs
- Caching for faster repeated calls
- Transparent integration with existing `by llm()` calls in Jac

---

## Features

1. **Automatic Interception**  
   - Every `by llm()` call in Jac is intercepted by Skynet before the original MTLLM plugin.
   - CLI logs indicate which path is taken (cache, local SLM, or API).

2. **Dynamic Model Switching**  
   - Local SLMs are used for short prompts or cached data.
   - API LLMs are used as a fallback or for complex prompts.
   - Training on local SLMs happens automatically from API interactions.

3. **GPU Device Detection**  
   - Detects NVIDIA or AMD GPUs and selects the best backend.
   - Falls back to CPU if no GPU is available.

4. **Caching**  
   - Responses are cached automatically for repeated prompts.
   - Improves speed and reduces API calls.

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd skynet-mtllm

# Install in editable mode
pip install --upgrade pip setuptools wheel
pip install -e .

# Optional Dependancies
pip install torch transformers sentence-transformers

# Skynet MTLLM — Changing the Local SLM

Skynet MTLLM allows you to dynamically choose which local small LLM (SLM) to use for prompt handling and incremental training. By default, the plugin uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, but you can configure any Hugging Face model installed locally or available via the Transformers library.

---

# Skynet MTLLM — Environment Variables and Local LLM Configuration

Skynet MTLLM can be configured using environment variables to control the behavior of local small language models (SLMs), API usage, caching, and device selection. Setting these variables allows you to customize how the plugin intercepts `by llm()` calls and gradually shifts to a local model.

---

## 1. Environment Variables

The local SLM and plugin behavior are controlled via the following environment variables:

| Variable                | Default                               | Description                                                                                                                         |
|-------------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `SKYNET_SLM_BASE`       | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  | Hugging Face model ID for the local SLM or file path for trained downloaded ones. The model is trained incrementally using API
responses. |
| `SKYNET_SIM_THRESHOLD`  | `0.72`                                | Similarity threshold for switching from API to local SLM. Once local SLM similarity exceeds this value, the plugin starts using it automatically. |
| `SKYNET_CACHE_DIR`      | `./skynet_cache`                      | Directory to store cached prompt-response pairs and incremental local SLM training data.                                            |
| `SKYNET_DEVICE`         | Auto (`cuda`/`cpu`)                   | Device to use for local SLM inference and training (`cpu`, `cuda`, `mps`, `rocm`).                                                  |
| `SKYNET_USE_API`        | `true`                                | Enable (`true`) or disable (`false`) API calls. Useful for offline mode or testing local models.                                    |
| `SKYNET_TRAIN_LOCAL`    | `true`                                | Enable (`true`) or disable (`false`) incremental training of the local SLM with API responses.                                      |

---

## 2. Usage Examples

### Windows PowerShell

```powershell
# Set local model to Tiny-LLM using Hugging Face model ID
$env:SKYNET_SLM_BASE="arnir0/Tiny-LLM"

# Set local model that has trained and downloaded
$env:SKYNET_SLM_BASE="C:\Users\User\Models\my_trained_model"

# Set similarity threshold for switching
$env:SKYNET_SIM_THRESHOLD="0.75"

# Specify cache directory
$env:SKYNET_CACHE_DIR="C:\Users\User\Documents\skynet_cache"

# Force device to CUDA (Can change to any but first need to install it sepratly)
$env:SKYNET_DEVICE="cuda"

# Enable API calls
$env:SKYNET_USE_API="true"

# Enable incremental training for local SLM
$env:SKYNET_TRAIN_LOCAL="true"

# Run Jac program
jac run main.jac




