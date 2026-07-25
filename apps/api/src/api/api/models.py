from concurrent.futures import thread
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union


class AgentRequest(BaseModel):
    query: str
    thread_id: str


class RAGUsedContext(BaseModel):
    image_url: str
    description: str
    price: Optional[float] = None  # sometimes the price is not available


class AgentResponse(BaseModel):
    answer: str
    used_context: list[RAGUsedContext]
    trace_id: str


class FeedbackRequest(BaseModel):
    trace_id: str
    feedback_score: Union[int, None] = Field(
        default=None,
        description="The score of the feedback, 0 for bad, 1 for good, None for no feedback",
    )
    feedback_text: str = Field(description="Feedback text")
    feedback_source_type: Literal["api", "model"] = Field(
        description="LangSmith feedback source type",
    )


class FeedbackResponse(BaseModel):
    message: str = Field(description="Submitted feedback")
