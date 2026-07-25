from pydantic import BaseModel, Field
from typing import Annotated, List, Any
from operator import add
from api.agents.agents import RAGUsedContext, agent_node, intent_router_node
from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langgraph.checkpoint.postgres import PostgresSaver
import json


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    final_answer: bool = False
    references: list[RAGUsedContext] = []
    trace_id: str = ""


### functions that control the Edges
def tool_router(state: State) -> str:

    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"


def intent_router_conditional_edges(state: State) -> str:

    if state.question_relevant:
        return "agent_node"
    else:
        return "end"


### Workflow

workflow = StateGraph(State)

tools = [get_formatted_item_context, get_formatted_reviews_context]
tool_node = ToolNode(tools)

workflow.add_node("tool_node", tool_node)
workflow.add_node("agent_node", agent_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")

workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {"agent_node": "agent_node", "end": END},
)

workflow.add_conditional_edges(
    "agent_node", tool_router, {"tools": "tool_node", "end": END}
)

workflow.add_edge("tool_node", "agent_node")

graph = workflow.compile()


### Agent Execution
def run_agent(question: str) -> dict:

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration": 0,
    }

    result = graph.invoke(initial_state)

    return result


def agent_stream_wrapper(question: str, thread_id: str) -> dict:
    # entire runction is a generator that yields a string for each event
    # get result and extract addtional data from qdrant database

    def _string_for_sse(string):
        return f"data: {string}\n\n"

    def _process_graph_event(chunk):
        """Convert a LangGraph stream event into a user-facing progress message.

        The function expects an event emitted by ``graph.stream`` when using
        ``stream_mode=["updates", "tasks"]``. A chunk is a ``(mode, event)``
        tuple. Only task-start events are handled; update and task-result events
        are ignored.

        Progress messages are printed for the intent router, agent, and tool
        nodes. Tool calls are translated into short descriptions of the work
        being performed.

        Args:
            chunk: A ``(stream_mode, event_data)`` tuple yielded by LangGraph.

        Returns:
            None. Progress information is written to standard output.
        """

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _tool_to_text(tool_call):
            if tool_call.get("name") == "get_formatted_item_context":
                return f"Looking for items: {tool_call.get('args').get('query', '')}."
            elif tool_call.get("name") == "get_formatted_reviews_context":
                return f"Fetching user reviews..."

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "intent_router_node":
                return "Analysing the question..."
            if chunk[1].get("payload", {}).get("name") == "agent_node":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "tool_node":
                message = " ".join(
                    [
                        _tool_to_text(tool_call)
                        for tool_call in chunk[1]
                        .get("payload", {})
                        .get("input", {})
                        .messages[-1]
                        .tool_calls
                    ]
                )
                return message

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}  # random id

    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"  # changed to service account instead of local
    ) as checkpointer:

        graph = workflow.compile(checkpointer=checkpointer)

        result = {}
        for event in graph.stream(
            initial_state, config, stream_mode=["debug", "values"]
        ):  # stream return a generator
            processed_chunk = _process_graph_event(event)
            if processed_chunk:
                yield _string_for_sse(processed_chunk)
            if event[0] == "values":
                result = event[1]

    # result = run_agent(question)

    used_context = []

    for item in result.get("references", []):
        payload = qdrant_client.scroll(
            collection_name="Amazon-items-collection-01-hybrid-search",
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin", match=MatchValue(value=item.get("id"))
                    )
                ]
            ),
        )[0][0].payload
        image_url = payload.get("image", "")
        price = payload.get("price")
        if image_url:
            used_context.append(
                {
                    "image_url": image_url,
                    "price": price,
                    "description": item.get("description"),
                }
            )

    # convert the result to a string and yield it with sse format
    yield _string_for_sse(
        json.dumps(
            {
                "type": "final_answer",
                "data": {
                    "answer": result.get("answer", ""),
                    "used_context": used_context,
                    "trace_id": result.get("trace_id", ""),
                },
            }
        )
    )
