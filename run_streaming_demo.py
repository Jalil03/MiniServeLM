from src.model_loader import load_model_and_tokenizer
from src.decoder import stream_decode_with_cache
from src.config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_PRESET, MODEL_NAME, PRESETS


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


def main():
    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)
    preset = PRESETS[DEFAULT_PRESET]

    user_message = (
        "Explain KV cache in Transformer inference in simple words. "
        "Use a short explanation."
    )

    prompt = build_chat_prompt(tokenizer, user_message)

    print("=" * 80)
    print("MiniServeLM Streaming Demo")
    print("=" * 80)
    print(f"Device: {device}")
    print("-" * 80)
    print("Assistant:", end=" ", flush=True)

    for text_piece in stream_decode_with_cache(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=preset["temperature"],
        top_k=preset["top_k"],
        top_p=preset["top_p"],
    ):
        print(text_piece, end="", flush=True)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
