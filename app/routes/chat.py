import time
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.inference.loader import get_model_id, is_loaded, get_tokenizer
from app.inference.generate import (
    generate_text,
    chat_messages_to_prompt,
    chat_stream_sse,
)

chat_router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | str | None = None


@chat_router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    print(f"endpoint called: /v1/chat/completions")
    print(f"request: {request}")
    messages = [m.model_dump() for m in request.messages]
    stop = [request.stop] if isinstance(request.stop, str) else (request.stop or [])
    max_new_tokens = request.max_tokens or 256
    model_id = get_model_id()

    if request.stream:
        return StreamingResponse(
            chat_stream_sse(
                messages=messages,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    prompt = chat_messages_to_prompt(messages)
    text, finish_reason = generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=stop,
    )
    print(f"Generated text: {text}")

    tokenizer = get_tokenizer()
    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(text, allowed_special={'<|endoftext|>'}))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
