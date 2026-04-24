# MiniServeLM Benchmark Results

## Setup

- Model: HuggingFaceTB/SmolLM2-360M-Instruct
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- VRAM: 6 GB
- Backend: PyTorch + Hugging Face Transformers
- Decoding method: Manual greedy decoding
- Max new tokens: 80

## Goal

The goal of this benchmark is to compare manual autoregressive decoding with and without KV cache.

Two decoding modes were tested:

1. `no_cache`: the full sequence is passed to the model at every generation step.
2. `with_cache`: the prompt is processed once, then only the newest token is passed while reusing `past_key_values`.

## Results

| Target Prompt Tokens | Actual Prompt Tokens | No Cache Time | With Cache Time | No Cache Tok/s | With Cache Tok/s | Speedup |
|---------------------:|---------------------:|--------------:|----------------:|---------------:|-----------------:|--------:|
| 32   | 50   | 12.14s | 12.34s | 6.59 | 6.48 | 0.98x |
| 128  | 145  | 11.21s | 11.32s | 7.13 | 7.07 | 0.99x |
| 512  | 525  | 11.30s | 9.46s  | 7.08 | 8.46 | 1.19x |
| 1024 | 1038 | 29.18s | 11.63s | 2.74 | 6.88 | 2.51x |

## Main Observation

KV cache does not provide a clear benefit for very short prompts because the sequence is small and the overhead of cache management can dominate.

However, as the prompt length increases, KV cache becomes much more useful.

At around 1024 prompt tokens, KV cache reduced generation time from 29.18 seconds to 11.63 seconds, giving a speedup of 2.51x.

## Interpretation

Without KV cache, the model recomputes attention over the entire sequence at every new token.

With KV cache, the model reuses previously computed key/value tensors and only processes the newest token during generation.

This makes KV cache especially important for long-context inference.

## Conclusion

This benchmark shows that KV cache is not just a theoretical optimization. On a local RTX 3060 Laptop GPU with 6 GB VRAM, it provides a clear decoding speedup at longer context lengths.

The strongest result was:

> KV cache achieved a 2.51x speedup at 1038 prompt tokens using SmolLM2-360M-Instruct.