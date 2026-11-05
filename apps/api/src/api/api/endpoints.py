from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from api.api.models import (
    AgentRequest,
    FeedbackRequest,
    FeedbackResponse,
    HitlRequest,
)
from api.agents.graph import agent_stream_wrapper
import logging
from api.api.processors.submit_feedback import submit_feedback

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

rag_router = APIRouter()
hitl_router = APIRouter()
feedback_router = APIRouter()


@rag_router.post("/")
def chat(payload: AgentRequest) -> StreamingResponse:
    return StreamingResponse(
        agent_stream_wrapper(payload.query, payload.thread_id, "initialise"),
        media_type="text/event-stream",
    )


@hitl_router.post("/")
def hitl(payload: HitlRequest) -> StreamingResponse:
    return StreamingResponse(
        agent_stream_wrapper(payload.approved, payload.thread_id, "hitl"),
        media_type="text/event-stream",
    )


@feedback_router.post("/")
def send_feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:
    submit_feedback(
        trace_id=payload.trace_id,
        feedback_score=payload.feedback_score,
        feedback_text=payload.feedback_text,
        feedback_source_type=payload.feedback_source_type,
    )
    return FeedbackResponse(
        message="Feedback submitted successfully",
    )


api_router = APIRouter()
api_router.include_router(
    rag_router, prefix="/agent", tags=["agent"]
)  # change from rag to agent to make more inline with the update
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["feedback"])
api_router.include_router(
    hitl_router, prefix="/send_hitl_response", tags=["hitl"]
)
