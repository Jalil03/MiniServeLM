from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model_and_tokenizer(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device