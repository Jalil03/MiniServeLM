from src.model_loader import load_model_and_tokenizer
from src.decoder import greedy_decode_no_cache, greedy_decode_with_cache
from src.config import DEFAULT_MAX_NEW_TOKENS, MODEL_NAME


def build_chat_prompt(tokenizer, user_message: str) -> str:
    """
    Convert a normal user message into the chat format expected by an Instruct model.
    """

    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def print_result(result):
    print("=" * 80)
    print(f"Mode: {result['mode']}")
    print("-" * 80)
    print(result["output_text"])
    print("-" * 80)
    print(f"Generated tokens: {result['num_generated_tokens']}")
    print(f"Total time: {result['total_time_sec']:.4f} sec")

    if result["num_generated_tokens"] > 0:
        tps = result["num_generated_tokens"] / result["total_time_sec"]
        print(f"Approx. tokens/sec: {tps:.2f}")

    print("=" * 80)


def main():
    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)

    user_message = (
        "Explain the Transformer neural network architecture in simple words "
        "in 3 short sentences."
    )

    prompt = build_chat_prompt(tokenizer, user_message)

    print(f"Device: {device}")
    print("=" * 80)
    print("Formatted prompt sent to the model:")
    print("-" * 80)
    print(prompt)
    print("=" * 80)

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

    print_result(result_no_cache)
    print_result(result_with_cache)

    speedup = result_no_cache["total_time_sec"] / result_with_cache["total_time_sec"]

    print()
    print("SUMMARY")
    print("-" * 80)
    print(f"No cache time:   {result_no_cache['total_time_sec']:.4f} sec")
    print(f"With cache time: {result_with_cache['total_time_sec']:.4f} sec")
    print(f"Speedup:         {speedup:.2f}x")
    print("-" * 80)


if __name__ == "__main__":
    main()
