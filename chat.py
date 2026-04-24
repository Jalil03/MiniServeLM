from src.model_loader import load_model_and_tokenizer
from src.decoder import stream_decode_with_cache
from src.config import (
    MODEL_NAME,
    DEFAULT_MAX_HISTORY_MESSAGES,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PRESET,
    PRESETS,
)


def build_chat_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def trim_history(messages, max_history_messages: int):
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    non_system_messages = [msg for msg in messages if msg["role"] != "system"]

    trimmed_non_system = non_system_messages[-max_history_messages:]

    return system_messages + trimmed_non_system


def print_help():
    print()
    print("Commands:")
    print("  /exit             Quit the chat")
    print("  /help             Show commands")
    print("  /preset <name>    Change sampling preset")
    print()
    print("Available presets:")
    for preset_name in PRESETS:
        print(f"  {preset_name}")
    print()


def main():
    print("=" * 80)
    print("MiniServeLM CLI Chat")
    print("=" * 80)
    print(f"Loading model: {MODEL_NAME}")
    print("=" * 80)

    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)

    current_preset_name = DEFAULT_PRESET
    current_preset = PRESETS[current_preset_name]

    messages = [
        {
            "role": "system",
            "content": (
                "You are MiniServeLM, a helpful local AI assistant running on the user's machine. "
                "Answer clearly and directly. "
                "If the user asks who you are, say you are MiniServeLM, a local LLM inference demo."
            ),
        }
    ]

    print("=" * 80)
    print("MiniServeLM CLI Chat Ready")
    print("=" * 80)
    print(f"Model from config: {MODEL_NAME}")
    print(f"Loaded model:      {model.config._name_or_path}")
    print(f"Device:            {device}")
    print(f"Preset:            {current_preset_name}")
    print(f"Max new tokens:    {DEFAULT_MAX_NEW_TOKENS}")
    print(f"Max history msgs:  {DEFAULT_MAX_HISTORY_MESSAGES}")
    print("-" * 80)
    print("Type /help for commands. Type /exit to quit.")
    print("=" * 80)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "exit", "quit", "/quit"]:
            print("Goodbye.")
            break

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower().startswith("/preset"):
            parts = user_input.split()

            if len(parts) != 2:
                print("Usage: /preset <name>")
                print(f"Available presets: {', '.join(PRESETS)}")
                continue

            preset_name = parts[1].lower()

            if preset_name not in PRESETS:
                print(f"Unknown preset: {preset_name}")
                print(f"Available presets: {', '.join(PRESETS)}")
                continue

            current_preset_name = preset_name
            current_preset = PRESETS[current_preset_name]

            print(f"Preset changed to: {current_preset_name}")
            print(
                f"temperature={current_preset['temperature']}, "
                f"top_k={current_preset['top_k']}, "
                f"top_p={current_preset['top_p']}"
            )
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        messages = trim_history(messages, DEFAULT_MAX_HISTORY_MESSAGES)

        prompt = build_chat_prompt(tokenizer, messages)

        print("\nAssistant: ", end="", flush=True)

        assistant_response = ""

        for text_piece in stream_decode_with_cache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=current_preset["temperature"],
            top_k=current_preset["top_k"],
            top_p=current_preset["top_p"],
        ):
            print(text_piece, end="", flush=True)
            assistant_response += text_piece

        print()

        messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )


if __name__ == "__main__":
    main()