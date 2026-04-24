import csv
from pathlib import Path

from src.model_loader import load_model_and_tokenizer
from src.decoder import greedy_decode_no_cache, greedy_decode_with_cache
from src.config import DEFAULT_MAX_NEW_TOKENS, MODEL_NAME

TARGET_PROMPT_TOKENS = [32, 128, 512, 1024]

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "decode_benchmark.csv"


def build_chat_prompt(tokenizer, user_message: str) -> str:
    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def count_tokens(tokenizer, text: str) -> int:
    return tokenizer(text, return_tensors="pt")["input_ids"].shape[1]


def make_prompt_with_target_length(tokenizer, target_tokens: int):
    """
    Build a chat-formatted prompt with approximately target_tokens tokens.
    """

    base_sentence = (
        "Explain how Transformer neural networks process sequences using tokens, "
        "attention, hidden states, and prediction. "
    )

    user_message = ""

    while True:
        user_message += base_sentence

        prompt = build_chat_prompt(tokenizer, user_message)
        num_tokens = count_tokens(tokenizer, prompt)

        if num_tokens >= target_tokens:
            return prompt, num_tokens


def tokens_per_second(num_tokens: int, total_time: float) -> float:
    if total_time <= 0:
        return 0.0

    return num_tokens / total_time


def main():
    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)

    print("=" * 80)
    print("MiniServeLM Decode Benchmark")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Max new tokens: {DEFAULT_MAX_NEW_TOKENS}")
    print("=" * 80)

    rows = []

    # Small warmup so the first measured run is less noisy.
    warmup_prompt, _ = make_prompt_with_target_length(tokenizer, 32)

    _ = greedy_decode_with_cache(
        model=model,
        tokenizer=tokenizer,
        prompt=warmup_prompt,
        device=device,
        max_new_tokens=5,
    )

    _ = greedy_decode_no_cache(
        model=model,
        tokenizer=tokenizer,
        prompt=warmup_prompt,
        device=device,
        max_new_tokens=5,
    )

    for target_tokens in TARGET_PROMPT_TOKENS:
        prompt, actual_prompt_tokens = make_prompt_with_target_length(
            tokenizer=tokenizer,
            target_tokens=target_tokens,
        )

        print()
        print("-" * 80)
        print(f"Target prompt tokens: {target_tokens}")
        print(f"Actual prompt tokens: {actual_prompt_tokens}")
        print("-" * 80)

        result_no_cache = greedy_decode_no_cache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        )

        result_with_cache = greedy_decode_with_cache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        )

        no_cache_tps = tokens_per_second(
            result_no_cache["num_generated_tokens"],
            result_no_cache["total_time_sec"],
        )

        with_cache_tps = tokens_per_second(
            result_with_cache["num_generated_tokens"],
            result_with_cache["total_time_sec"],
        )

        speedup = (
            result_no_cache["total_time_sec"]
            / result_with_cache["total_time_sec"]
        )

        row = {
            "model": MODEL_NAME,
            "device": device,
            "target_prompt_tokens": target_tokens,
            "actual_prompt_tokens": actual_prompt_tokens,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "generated_tokens_no_cache": result_no_cache["num_generated_tokens"],
            "generated_tokens_with_cache": result_with_cache["num_generated_tokens"],
            "time_no_cache_sec": round(result_no_cache["total_time_sec"], 4),
            "time_with_cache_sec": round(result_with_cache["total_time_sec"], 4),
            "tokens_per_sec_no_cache": round(no_cache_tps, 2),
            "tokens_per_sec_with_cache": round(with_cache_tps, 2),
            "speedup": round(speedup, 2),
        }

        rows.append(row)

        print(f"No cache time:      {row['time_no_cache_sec']} sec")
        print(f"With cache time:    {row['time_with_cache_sec']} sec")
        print(f"No cache tok/sec:   {row['tokens_per_sec_no_cache']}")
        print(f"With cache tok/sec: {row['tokens_per_sec_with_cache']}")
        print(f"Speedup:            {row['speedup']}x")

    RESULTS_DIR.mkdir(exist_ok=True)

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print(f"Saved benchmark results to: {RESULTS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
