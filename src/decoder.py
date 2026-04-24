import time
import torch


def sync_if_cuda(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def apply_top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
):
    """
    Filter logits using top-k and/or top-p nucleus filtering.

    top_k:
        Keep only the k most likely tokens.

    top_p:
        Keep the smallest set of tokens whose cumulative probability
        is at least top_p.
    """

    filtered_logits = logits.clone()

    # Top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, filtered_logits.size(-1))

        top_k_values = torch.topk(filtered_logits, top_k, dim=-1).values
        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)

        filtered_logits = filtered_logits.masked_fill(
            filtered_logits < min_top_k_value,
            float("-inf"),
        )

    # Top-p filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered_logits,
            descending=True,
            dim=-1,
        )

        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep at least the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = torch.zeros_like(
            filtered_logits,
            dtype=torch.bool,
        )

        indices_to_remove.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )

        filtered_logits = filtered_logits.masked_fill(
            indices_to_remove,
            float("-inf"),
        )

    return filtered_logits


def sample_next_token(
    next_token_logits,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
):
    """
    Choose the next token using sampling.

    If temperature is 0 or negative, fallback to greedy decoding.
    """

    if temperature is None or temperature <= 0:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    scaled_logits = next_token_logits / temperature

    filtered_logits = apply_top_k_top_p_filtering(
        logits=scaled_logits,
        top_k=top_k,
        top_p=top_p,
    )

    probs = torch.softmax(filtered_logits, dim=-1)

    next_token_id = torch.multinomial(
        probs,
        num_samples=1,
    )

    return next_token_id


@torch.no_grad()
def sample_decode_with_cache(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
):
    """
    Manual sampling decoder WITH KV cache.

    This is closer to real LLM generation than greedy decoding because
    it can produce different outputs depending on the sampling settings.
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generated_ids = input_ids.clone()
    past_key_values = None
    next_token_id = None

    sync_if_cuda(device)
    start_time = time.perf_counter()

    for step in range(max_new_tokens):
        if step == 0:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        else:
            outputs = model(
                input_ids=next_token_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        next_token_logits = logits[:, -1, :]

        next_token_id = sample_next_token(
            next_token_logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    sync_if_cuda(device)
    total_time = time.perf_counter() - start_time

    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    generated_only_ids = generated_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True)

    return {
        "mode": "sample_with_cache",
        "prompt": prompt,
        "output_text": full_text,
        "generated_text": generated_text,
        "generated_ids": generated_ids,
        "total_time_sec": total_time,
        "num_generated_tokens": generated_ids.shape[1] - input_ids.shape[1],
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }

@torch.no_grad()
def greedy_decode_no_cache(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 50,
):
    """
    Manual greedy decoding WITHOUT KV cache.

    At each step, we send the full sequence again:
    prompt + all generated tokens.
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generated_ids = input_ids.clone()

    sync_if_cuda(device)
    start_time = time.perf_counter()

    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(generated_ids, device=device)

        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    sync_if_cuda(device)
    total_time = time.perf_counter() - start_time

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return {
        "mode": "no_cache",
        "prompt": prompt,
        "output_text": output_text,
        "generated_ids": generated_ids,
        "total_time_sec": total_time,
        "num_generated_tokens": generated_ids.shape[1] - input_ids.shape[1],
    }


@torch.no_grad()
def greedy_decode_with_cache(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 50,
):
    """
    Manual greedy decoding WITH KV cache.

    Step 1:
        Send the full prompt.

    Next steps:
        Send only the latest generated token.
        Reuse past_key_values.
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generated_ids = input_ids.clone()
    past_key_values = None
    next_token_id = None

    sync_if_cuda(device)
    start_time = time.perf_counter()

    for step in range(max_new_tokens):
        if step == 0:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        else:
            outputs = model(
                input_ids=next_token_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    sync_if_cuda(device)
    total_time = time.perf_counter() - start_time

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return {
        "mode": "with_cache",
        "prompt": prompt,
        "output_text": output_text,
        "generated_ids": generated_ids,
        "total_time_sec": total_time,
        "num_generated_tokens": generated_ids.shape[1] - input_ids.shape[1],
    }


## Adding streaming decoding 

@torch.no_grad()
def stream_decode_with_cache(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
):
    """
    Manual streaming decoder WITH KV cache.

    Instead of returning the full text at the end, this yields small text pieces
    as soon as new tokens are generated.
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generated_ids = input_ids.clone()
    past_key_values = None
    next_token_id = None

    previous_text = ""

    for step in range(max_new_tokens):
        if step == 0:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        else:
            outputs = model(
                input_ids=next_token_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        next_token_logits = logits[:, -1, :]

        next_token_id = sample_next_token(
            next_token_logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        generated_only_ids = generated_ids[0][input_ids.shape[1]:]
        current_text = tokenizer.decode(
            generated_only_ids,
            skip_special_tokens=True,
        )

        new_text = current_text[len(previous_text):]
        previous_text = current_text

        if new_text:
            yield new_text

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        if next_token_id.item() == tokenizer.eos_token_id:
            break