import time
from fastapi import APIRouter
from pydantic import BaseModel

from app.inference.loader import get_model_id, is_loaded

models_router = APIRouter()


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "custom"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


@models_router.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models — required by OpenWebUI."""
    models = []
    if is_loaded():
        models.append(ModelCard(id=get_model_id(), created=int(time.time()), owned_by="custom"))
    return ModelList(data=models)
