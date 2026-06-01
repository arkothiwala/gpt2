import time
import uuid
import json
from typing import Generator, Iterator
import torch
import torch.nn.functional as F

from app.inference.loader import get_model, get_tokenizer, get_device, get_model_id


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature == 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # remove tokens where cumulative prob exceeds top_p
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _hits_stop(text: str, stop: list[str]) -> str | None:
    """Return the stop sequence that was hit, or None."""
    for s in stop:
        if s in text:
            return s
    return None


def generate_text(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
    stop: list[str] | None = None,
) -> tuple[str, str]:
    """
    Generate text from a prompt (non-streaming).
    Returns (generated_text, finish_reason).
    """
    print("generate_text")
    print(f"prompt: {prompt}")
    print(f"temperature: {temperature}, top_p: {top_p}")
    model = get_model()
    tokenizer = get_tokenizer()
    device = get_device()

    stop = stop or []
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated_ids: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = input_ids[:, -model.context_length:]
            logits = model(window)
            next_logits = logits[0, -1, :]
            next_id = _sample_next_token(next_logits, temperature, top_p)
            generated_ids.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

            current_text = tokenizer.decode(generated_ids)
            hit = _hits_stop(current_text, stop)
            if hit:
                return current_text[: current_text.find(hit)], "stop"

    return tokenizer.decode(generated_ids), "length"


def stream_text(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
) -> Iterator[int]:
    """
    Yield token IDs one at a time for streaming.
    Stops at stop sequences or max_new_tokens.
    """
    print("stream_text")
    print(f"prompt: {prompt}")
    print(f"temperature: {temperature}, top_p: {top_p}")
    model = get_model()
    tokenizer = get_tokenizer()
    device = get_device()

    stop = stop or []
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated_ids: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = input_ids[:, -model.context_length:]
            logits = model(window)
            next_logits = logits[0, -1, :]
            next_id = _sample_next_token(next_logits, temperature, top_p)
            generated_ids.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
            yield next_id

            current_text = tokenizer.decode(generated_ids)
            if _hits_stop(current_text, stop):
                return


def chat_messages_to_prompt(messages: list[dict]) -> str:
    """Convert OpenAI-style messages to a plain text prompt."""
    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        lines.append(f"{role}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


# ── SSE helpers ──────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def completion_stream_sse(
    prompt: str,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
) -> Generator[str, None, None]:
    tokenizer = get_tokenizer()
    req_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    finish_reason = "length"

    generated_ids: list[int] = []
    for token_id in stream_text(prompt, max_new_tokens, temperature, top_p, stop):
        generated_ids.append(token_id)
        token_text = tokenizer.decode([token_id])
        chunk = {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "choices": [{"text": token_text, "index": 0, "logprobs": None, "finish_reason": None}],
        }
        yield _sse(chunk)

        current_text = tokenizer.decode(generated_ids)
        if stop and _hits_stop(current_text, stop):
            finish_reason = "stop"
            break

    final = {
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": finish_reason}],
    }
    yield _sse(final)
    yield "data: [DONE]\n\n"


def chat_stream_sse(
    messages: list[dict],
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
) -> Generator[str, None, None]:
    tokenizer = get_tokenizer()
    prompt = chat_messages_to_prompt(messages)
    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    finish_reason = "length"

    print("chat_stream_sse")
    print(f"temperature: {temperature}, top_p: {top_p}")

    # Send role delta first
    yield _sse({
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })

    generated_ids: list[int] = []
    for token_id in stream_text(prompt, max_new_tokens, temperature, top_p, stop):
        generated_ids.append(token_id)
        token_text = tokenizer.decode([token_id])
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
        }
        yield _sse(chunk)

        current_text = tokenizer.decode(generated_ids)
        if stop and _hits_stop(current_text, stop):
            finish_reason = "stop"
            break

    yield _sse({
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    })
    yield "data: [DONE]\n\n"
