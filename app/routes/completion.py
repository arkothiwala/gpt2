import time
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.inference.loader import get_model_id, is_loaded, get_tokenizer
from app.inference.generate import generate_text, completion_stream_sse

completion_router = APIRouter()


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int | None = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | str | None = ['<|endoftext|>']


@completion_router.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible text completions endpoint."""
    print(f"endpoint called: /v1/completions")
    print(f"request: {request}")
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    stop = [request.stop] if isinstance(request.stop, str) else (request.stop or [])
    max_new_tokens = request.max_tokens or 256
    model_id = get_model_id()

    if request.stream:
        return StreamingResponse(
            completion_stream_sse(
                prompt=prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text, finish_reason = generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=stop,
    )

    tokenizer = get_tokenizer()
    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(text))

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
