import os
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yaml
from app.routes.chat import chat_router
from app.routes.completion import completion_router
from app.routes.health import health_check
from app.routes.models import models_router
from app.inference.loader import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_device() -> str:
    raw = os.getenv("DEVICE", "auto")
    if raw == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return raw


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config_path = os.getenv("CONFIG_PATH", "configs/gpt2.yaml")
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if not checkpoint_path:
        checkpoint_path = yaml.safe_load(open(config_path)).get("checkpoint", {}).get("path") if os.path.exists(config_path) else None

    if checkpoint_path:
        device = _resolve_device()
        logger.info(f"Loading model: checkpoint={checkpoint_path} config={config_path} device={device}")
        load_model(checkpoint_path=checkpoint_path, config_path=config_path, device=device)
    else:
        logger.warning(
            "CHECKPOINT_PATH not set — model will not be loaded. "
            "Set CHECKPOINT_PATH=<path/to/checkpoint.pt> before starting."
        )

    yield  # server is running

    logger.info("Shutting down")


app = FastAPI(
    debug=False,
    title="LLM OpenAI API custom endpoint",
    summary="A custom endpoint for LLM OpenAI API mainly to test GPT2 style or other LLMs.",
    description=(
        "I want to learn to reverse engineer what vLLM is doing behind the scenes — "
        "different decoding strategies, optimizations, KV cache, Paged Attention etc. "
        "So I want to build a custom endpoint for LLM OpenAI API and test it with different LLMs."
    ),
    version="0.1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "ashutosh", "email": "gjak4u@gmail.com"},
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_check)
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(completion_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
