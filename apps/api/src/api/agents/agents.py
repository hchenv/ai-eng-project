import instructor

from langsmith import traceable, get_current_run_tree
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, convert_to_openai_messages, AIMessage
from langchain_openai import ChatOpenAI

from api.agents.utils.prompt_management import prompt_template_config

from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context


### QnA Agent Response Model


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(
        description="Description of the item used to answer the question"
    )


class FinalResponse(BaseModel):
    """Call this tool when the final answer is possible using available context."""

    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(
        description="List of items used to answer the question"
    )


### Intent Router Response Model


class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str = Field(
        description="An answer to the question if the users question is not relevant to the products."
    )


### QnA Agent Node


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-5.4-mini"},
)
def agent_node(state) -> dict:

    template = prompt_template_config("api/agents/prompts/qna_agent.yaml", "qna_agent")

    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini", reasoning_effort="low", use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [get_formatted_item_context, get_formatted_reviews_context, FinalResponse],
        tool_choice="required",  # why not "any", "any" works with completion api(langchain convert to "required"),not sure any works with response api
    )

    response = llm_with_tools.invoke([SystemMessage(content=prompt), *state.messages])

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage_metadata[
                "input_tokens"
            ],  # changed from raw_response.usage.input_tokens since we switched from instructor to langchain
            "output_tokens": response.usage_metadata["output_tokens"],
            "total_tokens": response.usage_metadata["total_tokens"],
        }

    final_answer = False
    answer = ""
    references = []

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalResponse":
                final_answer = True
                answer = tool_call.get("args").get("answer")
                references.extend(tool_call.get("args").get("references"))

                # Strip the tool_calls off the terminal turn so the message
                # persisted in state is a plain assistant message. A raw
                # tool_calls message (with no matching ToolMessage) is invalid
                # as input to the Responses API when the thread is replayed.
                response = AIMessage(content=answer)

    return {
        "messages": [response],
        "final_answer": final_answer,
        "iteration": state.iteration + 1,
        "answer": answer,
        "references": references,
    }


### Intent Router Node


@traceable(
    name="route_intent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-5.4-mini"},
)
def intent_router_node(state) -> dict:

    template = prompt_template_config(
        "api/agents/prompts/intent_router_agent.yaml", "intent_router_agent"
    )

    prompt = template.render()

    messages = state.messages

    conversation = []

    # for message in messages:
    #     conversation.append(convert_to_openai_messages(message))
    conversation.append(convert_to_openai_messages(messages[-1]))

    client = instructor.from_provider(
        "openai/gpt-5.4-mini", mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, raw_response = client.create_with_completion(
        messages=[{"role": "system", "content": prompt}, *conversation],
        reasoning={"effort": "none"},
        response_model=IntentRouterResponse,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        trace_id = str(current_run.trace_id)
    else:
        trace_id = ""

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
        "trace_id": trace_id,
    }
