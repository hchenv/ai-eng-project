from fastapi import APIRouter, Request
from api.api.models import RAGRequest, RAGResponse
from api.agents.retrieval_generation import rag_pipeline
import logging
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

rag_router = APIRouter()

qdrant_client = QdrantClient(url="http://qdrant:6333")

@rag_router.post("/")
def chat(payload: RAGRequest) -> RAGResponse:
    result =  rag_pipeline(payload.query, qdrant_client)
    return RAGResponse(answer=result['answer'])


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])