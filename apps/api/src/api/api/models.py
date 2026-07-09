from pydantic import BaseModel
from typing import Optional


class RAGRequest(BaseModel):
    query: str


class RAGUsedContext(BaseModel):
    image_url: str
    description: str
    price: Optional[float] = None  # sometimes the price is not available


class RAGResponse(BaseModel):
    answer: str
    used_context: list[RAGUsedContext]
