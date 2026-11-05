from pydantic import BaseModel, Field
from typing import Annotated, Any, List, Literal
from operator import add
from api.agents.agents import (
    RAGUsedContext,
    product_qna_agent,
    shopping_cart_agent,
    warehouse_manager_agent,
    coordinator_agent,
    MAX_ITERATIONS_PRODUCT_QNA,
    MAX_ITERATIONS_SHOPPING_CART,
    MAX_ITERATIONS_WAREHOUSE_MANAGER,
    MAX_ITERATIONS_COORDINATOR,
)
from api.agents.tools import (
    get_formatted_item_context,
    get_formatted_reviews_context,
    get_shopping_cart,
    get_shopping_cart_for_sse,
    remove_from_cart,
    add_to_shopping_cart,
    check_warehouse_availability,
    reserve_warehouse_items,
    get_qdrant_client,
)
from api.agents.guardrails import redact_product_id
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langgraph.checkpoint.postgres import PostgresSaver
from api.core.config import config
import json


class AgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    messages: List[Any] = Field(default_factory=list)


class CoordinatorAgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    next_agent: str = ""
    next_agent_task: str = ""


class State(BaseModel):
    messages: Annotated[List[Any], add] = Field(default_factory=list)
    user_intent: str = ""
    product_qna_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    warehouse_manager_agent: AgentProperties = Field(default_factory=AgentProperties)
    coordinator_agent: CoordinatorAgentProperties = Field(
        default_factory=CoordinatorAgentProperties
    )
    answer: str = ""
    references: list[RAGUsedContext] = Field(default_factory=list)
    user_id: str = ""
    cart_id: str = ""
    trace_id: str = ""


### Edges


def product_qna_agent_tool_router(state) -> str:

    if state.product_qna_agent.final_answer:
        return "end"
    elif state.product_qna_agent.iteration > MAX_ITERATIONS_PRODUCT_QNA:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"


def shopping_cart_agent_tool_router(state) -> str:

    add_to_cart_tool_call = False
    for tool_call in state.shopping_cart_agent.messages[-1].tool_calls:
        if tool_call.get("name") == "add_to_shopping_cart":
            add_to_cart_tool_call = True
            break

    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > MAX_ITERATIONS_SHOPPING_CART:
        return "end"
    elif len(state.shopping_cart_agent.messages[-1].tool_calls) > 0:
        if add_to_cart_tool_call:
            return "hitl_add_to_cart"
        else:
            return "tools"
    else:
        return "end"


def warehouse_manager_agent_tool_router(state) -> str:

    if state.warehouse_manager_agent.final_answer:
        return "end"
    elif state.warehouse_manager_agent.iteration > MAX_ITERATIONS_WAREHOUSE_MANAGER:
        return "end"
    elif len(state.warehouse_manager_agent.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"


def coordinator_agent_edge(state) -> str:

    if state.coordinator_agent.final_answer:
        return "end"
    elif state.coordinator_agent.iteration > MAX_ITERATIONS_COORDINATOR:
        return "end"
    elif state.coordinator_agent.next_agent == "product_qna_agent":
        return "product_qna_agent"
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    elif state.coordinator_agent.next_agent == "warehouse_manager_agent":
        return "warehouse_manager_agent"
    else:
        return "end"


### Human In The Loop Node


def hitl_add_to_cart(state) -> Command[Literal["shopping_cart_agent_tool_node", END]]:

    for tool_call in state.shopping_cart_agent.messages[-1].tool_calls:
        if tool_call.get("name") == "add_to_shopping_cart":
            items_to_add = tool_call.get("args").get("items")
            break

    human_input = interrupt({"items_to_add": items_to_add})

    ### ALWAYS PAUSES HERE

    if human_input.get("confirmed"):
        return Command(update={}, goto="shopping_cart_agent_tool_node")
    else:
        last_msg = state.shopping_cart_agent.messages[-1]
        sanitized_msg = AIMessage(content=last_msg.content, id=last_msg.id)

        return Command(
            update={
                "shopping_cart_agent": {"messages": [sanitized_msg]},
                "answer": "You have rejected the addition of items to the cart.",
            },
            goto=END,
        )


### Edges

### Workflow

workflow = StateGraph(State)

product_qna_agent_tools = [get_formatted_item_context, get_formatted_reviews_context]
product_qna_agent_tool_node = ToolNode(product_qna_agent_tools)

shopping_cart_agent_tools = [get_shopping_cart, remove_from_cart, add_to_shopping_cart]
_shopping_cart_agent_tool_node = ToolNode(shopping_cart_agent_tools)


def shopping_cart_agent_tool_node(state):
    agent_state = state.shopping_cart_agent
    output = _shopping_cart_agent_tool_node.invoke({"messages": agent_state.messages})

    return {
        "shopping_cart_agent": agent_state.model_copy(
            update={"messages": agent_state.messages + output["messages"]}
        )
    }


warehouse_manager_agent_tools = [check_warehouse_availability, reserve_warehouse_items]
_warehouse_manager_agent_tool_node = ToolNode(warehouse_manager_agent_tools)


def warehouse_manager_agent_tool_node(state):
    agent_state = state.warehouse_manager_agent
    output = _warehouse_manager_agent_tool_node.invoke(
        {"messages": agent_state.messages}
    )

    return {
        "warehouse_manager_agent": agent_state.model_copy(
            update={"messages": agent_state.messages + output["messages"]}
        )
    }


workflow.add_node("product_qna_agent_tool_node", product_qna_agent_tool_node)
workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)
workflow.add_node(
    "warehouse_manager_agent_tool_node", warehouse_manager_agent_tool_node
)
workflow.add_node("product_qna_agent", product_qna_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("warehouse_manager_agent", warehouse_manager_agent)
workflow.add_node("coordinator_agent", coordinator_agent)
workflow.add_node("hitl_add_to_cart", hitl_add_to_cart)
workflow.add_node("redact_product_id", redact_product_id)

workflow.add_edge(START, "coordinator_agent")

workflow.add_conditional_edges(
    "coordinator_agent",
    coordinator_agent_edge,
    {
        "product_qna_agent": "product_qna_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "warehouse_manager_agent": "warehouse_manager_agent",
        "end": "redact_product_id",
    },
)

workflow.add_conditional_edges(
    "product_qna_agent",
    product_qna_agent_tool_router,
    {"tools": "product_qna_agent_tool_node", "end": "coordinator_agent"},
)

workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_router,
    {
        "tools": "shopping_cart_agent_tool_node",
        "hitl_add_to_cart": "hitl_add_to_cart",
        "end": "coordinator_agent",
    },
)

workflow.add_conditional_edges(
    "warehouse_manager_agent",
    warehouse_manager_agent_tool_router,
    {"tools": "warehouse_manager_agent_tool_node", "end": "coordinator_agent"},
)

workflow.add_edge("product_qna_agent_tool_node", "product_qna_agent")
workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")
workflow.add_edge("warehouse_manager_agent_tool_node", "warehouse_manager_agent")
workflow.add_edge("redact_product_id", END)


### Agent Execution


def agent_stream_wrapper(question, thread_id: str, mode: str = "initialise"):

    def _string_for_sse(string):
        return f"data: {string}\n\n"

    def _process_graph_event(chunk):

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _is_interrupt(chunk):
            return len(chunk[1].get("payload", {}).get("interrupts", [])) > 0

        def _tool_to_text(tool_call):
            if tool_call.get("name") == "get_formatted_item_context":
                return f"Looking for items: {tool_call.get('args').get('query', '')}."
            elif tool_call.get("name") == "get_formatted_reviews_context":
                return f"Fetching user reviews..."
            elif tool_call.get("name") == "get_shopping_cart":
                return "Fetching shopping cart..."
            elif tool_call.get("name") == "remove_from_cart":
                return "Removing items from shopping cart..."
            elif tool_call.get("name") == "add_to_shopping_cart":
                return "Adding items to shopping cart..."
            elif tool_call.get("name") == "check_warehouse_availability":
                return "Checking warehouse availability..."
            elif tool_call.get("name") == "reserve_warehouse_items":
                return "Reserving warehouse items..."

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "coordinator_agent":
                return "Analysing the question..."
            if chunk[1].get("payload", {}).get("name") == "product_qna_agent":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "shopping_cart_agent":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "warehouse_manager_agent":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name", "").endswith("tool_node"):
                node_name = chunk[1].get("payload", {}).get("name")
                node_input = chunk[1].get("payload", {}).get("input")

                if node_name == "shopping_cart_agent_tool_node":
                    tool_calls = node_input.shopping_cart_agent.messages[-1].tool_calls
                elif node_name == "warehouse_manager_agent_tool_node":
                    tool_calls = node_input.warehouse_manager_agent.messages[
                        -1
                    ].tool_calls
                else:
                    tool_calls = node_input.messages[-1].tool_calls

                message = " ".join(
                    [_tool_to_text(tool_call) for tool_call in tool_calls]
                )
                return message
        elif _is_interrupt(chunk):
            value = chunk[1].get("payload", {}).get("interrupts", [])[0].get("value")
            payload = {
                "type": "hitl_interrupt",
                "data": {
                    "data": value,
                },
            }
            return json.dumps(payload)

    qdrant_client = get_qdrant_client()

    if mode == "initialise":
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "user_id": thread_id,
            "cart_id": thread_id,
            "coordinator_agent": {
                "iteration": 0,
                "final_answer": False,
                "next_agent": "",
                "next_agent_task": "",
            },
            "product_qna_agent": {
                "iteration": 0,
                "final_answer": False,
                "messages": [],
            },
            "shopping_cart_agent": {
                "iteration": 0,
                "final_answer": False,
                "messages": [],
            },
            "warehouse_manager_agent": {
                "iteration": 0,
                "final_answer": False,
                "messages": [],
            },
        }
    elif mode == "hitl":
        initial_state = Command(
            resume={
                "confirmed": question,
            }
        )
    else:
        raise ValueError(f"Unsupported agent stream mode: {mode}")

    graph_config = {"configurable": {"thread_id": thread_id}}
    result = {}
    interrupted = False

    with PostgresSaver.from_conn_string(
        config.LANGGRAPH_DB_URL
    ) as checkpointer:

        graph = workflow.compile(checkpointer=checkpointer)

        for chunk in graph.stream(
            initial_state, config=graph_config, stream_mode=["debug", "values"]
        ):

            processed_chunk = _process_graph_event(chunk)

            if processed_chunk:
                try:
                    parsed_chunk = json.loads(processed_chunk)
                    if parsed_chunk.get("type") == "hitl_interrupt":
                        interrupted = True
                except (TypeError, json.JSONDecodeError):
                    pass
                yield _string_for_sse(processed_chunk)

            if chunk[0] == "values":
                result = chunk[1]

    if interrupted:
        return

    used_context = []

    for item in result.get("references", []):
        points, _ = qdrant_client.scroll(
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
            limit=1,
        )
        if not points:
            continue
        payload = points[0].payload
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

    shopping_cart = get_shopping_cart_for_sse(user_id=thread_id, cart_id=thread_id)
    shopping_cart_items = [
        {
            "price": float(item.get("price")) if item.get("price") else None,
            "quantity": item.get("quantity"),
            "currency": item.get("currency"),
            "product_image_url": item.get("product_image_url"),
            "total_price": (
                float(item.get("total_price")) if item.get("total_price") else None
            ),
        }
        for item in shopping_cart
    ]

    yield _string_for_sse(
        json.dumps(
            {
                "type": "final_answer",
                "data": {
                    "answer": result.get("answer", ""),
                    "used_context": used_context,
                    "trace_id": result.get("trace_id", ""),
                    "shopping_cart": shopping_cart_items,
                },
            }
        )
    )
