from src.model_loader import load_model_and_tokenizer
from src.decoder import sample_decode_with_cache
from src.config import DEFAULT_MAX_NEW_TOKENS, MODEL_NAME, PRESETS


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


def print_result(title, result):
    print("=" * 80)
    print(title)
    print("-" * 80)
    print(f"Temperature: {result['temperature']}")
    print(f"Top-k:       {result['top_k']}")
    print(f"Top-p:       {result['top_p']}")
    print("-" * 80)
    print(result["generated_text"])
    print("-" * 80)
    print(f"Generated tokens: {result['num_generated_tokens']}")
    print(f"Total time:       {result['total_time_sec']:.4f} sec")

    if result["num_generated_tokens"] > 0:
        tps = result["num_generated_tokens"] / result["total_time_sec"]
        print(f"Tokens/sec:       {tps:.2f}")

    print("=" * 80)


def main():
    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)

    user_message = (
        "Explain the Transformer neural network architecture in simple words. "
        "Focus on tokens, attention, and prediction."
    )

    prompt = build_chat_prompt(tokenizer, user_message)

    print(f"Device: {device}")

    for preset_name, preset in PRESETS.items():
        result = sample_decode_with_cache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=preset["temperature"],
            top_k=preset["top_k"],
            top_p=preset["top_p"],
        )

        print_result(f"{preset_name.title()} Sampling", result)


if __name__ == "__main__":
    main()
