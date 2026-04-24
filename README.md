# MiniServeLM — A Minimal Local LLM Inference Engine

MiniServeLM is a small local LLM inference engine built to understand how text generation works under the hood.

Instead of only calling `model.generate()`, this project manually implements the core generation loop:

```text
prompt
→ tokenizer
→ input_ids
→ model forward pass
→ logits
→ choose next token
→ append token
→ repeat
```

The project supports manual decoding, KV cache decoding, sampling, streaming generation, and a CLI chat interface.

---

## Why This Project Exists

This project is different from a pure benchmark.

A benchmark asks:

> How fast is KV cache compared to no cache?

MiniServeLM asks:

> How does LLM inference actually work, and can I build a small local inference engine myself?

The goal is to learn and implement the mechanics behind local LLM serving:

- tokenization
- manual autoregressive decoding
- logits processing
- greedy decoding
- KV cache decoding
- temperature sampling
- top-k sampling
- top-p sampling
- streaming generation
- CLI chat interaction

---

## Features

- Manual greedy decoding
- Manual KV-cache decoding using `past_key_values`
- Temperature sampling
- Top-k sampling
- Top-p / nucleus sampling
- Streaming token output
- CLI chat mode
- Conversation history trimming
- Generation presets
- Configurable model name and decoding settings
- Lightweight benchmark script for validating cache behavior

---

## Current Default Model

MiniServeLM currently uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

This model was chosen because it is stronger than very small models like SmolLM2-360M, while still being realistic on a 6 GB VRAM laptop GPU.

The default configuration is stored in:

```text
src/config.py
```

---

## Hardware Used

The project was tested locally on:

```text
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
VRAM: 6 GB
Backend: PyTorch + Hugging Face Transformers
OS: Windows
```

The model runs on CUDA when PyTorch detects the GPU.

---

## Project Structure

```text
MiniServeLM/
│
├── chat.py                  # CLI chat interface
├── run_manual_decode.py     # Compare manual no-cache vs KV-cache decoding
├── run_sampling_demo.py     # Demonstrate sampling presets
├── run_streaming_demo.py    # Demonstrate streaming generation
├── benchmark_decode.py      # Lightweight decoding benchmark
├── BENCHMARK_RESULTS.md     # Benchmark notes/results
├── requirements.txt
├── README.md
│
└── src/
    ├── __init__.py
    ├── config.py            # Model name and generation presets
    ├── decoder.py           # Manual decoding, sampling, streaming
    └── model_loader.py      # Model/tokenizer loading
```

---

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

If PyTorch was installed as CPU-only, install a CUDA-enabled PyTorch build:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Check that CUDA is visible:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Expected result:

```text
True
NVIDIA GeForce RTX 3060 Laptop GPU
```

---

## Run the CLI Chat

Start the local chat interface:

```powershell
python chat.py
```

Example output:

```text
MiniServeLM CLI Chat Ready
Model from config: Qwen/Qwen2.5-1.5B-Instruct
Loaded model:      Qwen/Qwen2.5-1.5B-Instruct
Device:            cuda
Preset:            balanced
```

Inside the chat, you can type normal messages:

```text
You: who are you?
Assistant: I am MiniServeLM, a local LLM inference demo.
```

---

## Chat Commands

```text
/help             Show commands
/exit             Quit the chat
/preset <name>    Change sampling preset
```

Available presets:

```text
factual
conservative
balanced
creative
```

Example:

```text
/preset factual
```

---

## Generation Presets

The generation presets are defined in:

```text
src/config.py
```

Current presets:

```python
PRESETS = {
    "factual": {
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
    },
    "conservative": {
        "temperature": 0.3,
        "top_k": 20,
        "top_p": 0.90,
    },
    "balanced": {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
    },
    "creative": {
        "temperature": 1.0,
        "top_k": 80,
        "top_p": 0.98,
    },
}
```

---

## Manual Decoding Demo

Run:

```powershell
python run_manual_decode.py
```

This compares two decoding modes:

```text
no_cache
with_cache
```

### No-cache decoding

The full sequence is passed to the model at every generation step.

```text
Step 1: prompt
Step 2: prompt + token_1
Step 3: prompt + token_1 + token_2
```

This is simple, but inefficient for long contexts.

### KV-cache decoding

The prompt is processed once. Then, each next step sends only the newest token while reusing `past_key_values`.

```text
Step 1: full prompt
Step 2: latest token + cached keys/values
Step 3: latest token + cached keys/values
```

This avoids recomputing previous attention keys and values.

---

## Sampling Demo

Run:

```powershell
python run_sampling_demo.py
```

This demonstrates different decoding styles:

```text
factual
conservative
balanced
creative
```

The sampling decoder supports:

- temperature
- top-k filtering
- top-p / nucleus filtering

---

## Streaming Demo

Run:

```powershell
python run_streaming_demo.py
```

This prints generated text progressively, token by token, instead of waiting for the full response at the end.

This is closer to how real assistant interfaces feel.

---

## Benchmark

Run the lightweight benchmark:

```powershell
python benchmark_decode.py
```

The benchmark compares:

```text
manual no-cache decoding
manual KV-cache decoding
```

across different prompt lengths.

The benchmark is intentionally lightweight because this project is designed to run safely on a 6 GB VRAM laptop GPU.

---

## Example Benchmark Result

Earlier tests with the smaller SmolLM2-360M-Instruct model showed the expected KV cache behavior:

| Prompt Tokens | No Cache Time | With Cache Time | Speedup |
|--------------:|--------------:|----------------:|--------:|
| 50            | 12.14s        | 12.34s          | 0.98x   |
| 145           | 11.21s        | 11.32s          | 0.99x   |
| 525           | 11.30s        | 9.46s           | 1.19x   |
| 1038          | 29.18s        | 11.63s          | 2.51x   |

Main observation:

> KV cache provides little benefit for short prompts, but becomes much more useful as context length increases.

This benchmark is used as a validation tool. The main identity of MiniServeLM is not benchmarking, but building the inference engine.

---

## Important Notes

### Model quality

The default model is small enough to run locally on limited hardware, so it will not match the quality of large hosted models.

MiniServeLM focuses on inference mechanics, not state-of-the-art response quality.

### GPU temperature

Long benchmarks can push laptop GPUs hard. For safer local development:

- keep the laptop plugged in
- keep airflow clear
- avoid running heavy benchmarks repeatedly
- monitor GPU temperature with:

```powershell
nvidia-smi -l 1
```

---

## What I Learned

This project helped me understand:

- how causal language models generate one token at a time
- why logits are not text, but scores over the vocabulary
- how greedy decoding works
- how sampling changes model behavior
- why KV cache matters for long-context inference
- how streaming generation is implemented
- how a local CLI assistant can be built from a model and a manual decoder

---

## Roadmap

Planned improvements:

- add `/model` command in chat
- add `/settings` command in chat
- add better output formatting
- add optional model switching
- add FastAPI serving endpoint
- compare manual decoding with `model.generate()`
- add unit tests for sampling and decoding utilities
- clean benchmark scripts for repeatable experiments

---

## Repo Identity

MiniServeLM is best described as:

> A minimal local LLM inference engine with manual decoding, KV cache, sampling, streaming, and CLI chat.

It is designed to be understandable, hackable, and reproducible on consumer hardware.
