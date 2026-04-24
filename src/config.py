MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DEFAULT_MAX_NEW_TOKENS = 80
DEFAULT_MAX_HISTORY_MESSAGES = 4
DEFAULT_PRESET = "balanced"

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