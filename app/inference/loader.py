import os
import logging
import torch
import tiktoken
import yaml
from gpt.modules.models.gpt2 import GPT2Model

logger = logging.getLogger(__name__)

_model: GPT2Model | None = None
_tokenizer = None
_device: str = "cpu"
_model_id: str = "gpt2-custom"


def load_model(checkpoint_path: str, config_path: str, device: str = "cpu") -> None:
    global _model, _tokenizer, _device

    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    model = GPT2Model(
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        vocab_size=model_cfg["vocab_size"],
        context_length=model_cfg["context_length"],
    )

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _model = model
    _tokenizer = tiktoken.get_encoding("gpt2")
    _device = device
    logger.info(f"Model loaded on {device}")


def get_model() -> GPT2Model:
    if _model is None:
        raise RuntimeError("Model not loaded. Set CHECKPOINT_PATH and CONFIG_PATH env vars.")
    return _model


def get_tokenizer():
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not loaded.")
    return _tokenizer


def get_device() -> str:
    return _device


def get_model_id() -> str:
    return _model_id


def is_loaded() -> bool:
    return _model is not None
