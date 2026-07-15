import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import openai
from langsmith import traceable, get_current_run_tree

# for cost calculation
from qdrant_client import QdrantClient
from qdrant_client.http.models.models import Payload
from api.core.config import config
import instructor
from pydantic import BaseModel, Field
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, Document
from qdrant_client import models
from api.agents.utils.prompt_management import prompt_template_config
import cohere


class RAGUsedContext(BaseModel):
    id: str = Field(
        description="ID of the item used to answer the question"
    )  # only id needed since we can query qdrant for the rest
    description: str = Field(
        description="Description of the item used to answer the question"
    )  # why need description? i want to show a summary to the user because we do not have in the vector db


class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(
        description="List of items used to the context"
    )


qdrant_client = QdrantClient(
    url="http://qdrant:6333"
)  # on docker image it called qdrant as defined


@traceable(
    name="get_embedding",
    run_type="embedding",
    metadata={"ls_model_name": "text-embedding-3-small", "ls_provider": "openai"},
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return response.data[0].embedding


@traceable(name="retrieve_from_qdrant", run_type="retriever")
def retrieve_from_qdrant(query, qdrant_client, k=5, hybrid=True):
    # response = qdrant_client.query_points(
    #     collection_name="amazon-electronics-items-collection-01",
    #     query=get_embedding(query),
    #     limit=k,
    # )
    query_embedding = get_embedding(query)
    if hybrid:
        response = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using="text-embedding-3-small",
                    limit=20,  # why 20? limit is often 3x - 10x of k
                ),
                Prefetch(
                    query=Document(
                        text=query,
                        model="qdrant/bm25",  # bm search need actual query text
                    ),
                    using="bm25",
                    limit=20,
                ),
            ],
            query=models.RrfQuery(
                rrf=models.Rrf(weights=[3, 1])
            ),  # should be tuned, based on the data; contextual vs keyword search
            limit=k,
        )
    else:
        response = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            query=query_embedding,
            using="text-embedding-3-small",
            limit=k,
        )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []
    for point in response.points:
        retrieved_context_ids.append(point.payload["parent_asin"])
        retrieved_context.append(point.payload["preprocessed_description"])
        similarity_scores.append(point.score)
        retrieved_context_ratings.append(point.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings,
    }


@traceable(name="rerank_data", run_type="tool")
def rerank_data(query, context, top_k=5):

    cohere_client = cohere.ClientV2()

    response = cohere_client.rerank(
        model="rerank-v4.0-pro",
        query=query,
        documents=context["retrieved_context"],
        top_n=top_k,
    )

    order = [result.index for result in response.results]

    return {
        "retrieved_context_ids": [context["retrieved_context_ids"][i] for i in order],
        "retrieved_context": [context["retrieved_context"][i] for i in order],
        "similarity_scores": [context["similarity_scores"][i] for i in order],
        "retrieved_context_ratings": [
            context["retrieved_context_ratings"][i] for i in order
        ],
    }


@traceable(name="rerank_data_openrouter", run_type="tool")
def rerank_data_openrouter(query, context, top_k=5, model="cohere/rerank-4-pro"):
    request_body = {
        "model": model,
        "query": query,
        "documents": context["retrieved_context"],
        "top_n": top_k,
    }

    request = Request(
        "https://openrouter.ai/api/v1/rerank",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request) as response:
        response_body = json.loads(response.read().decode("utf-8"))

    order = [result["index"] for result in response_body["results"]]

    return {
        "retrieved_context_ids": [context["retrieved_context_ids"][i] for i in order],
        "retrieved_context": [context["retrieved_context"][i] for i in order],
        "similarity_scores": [context["similarity_scores"][i] for i in order],
        "retrieved_context_ratings": [
            context["retrieved_context_ratings"][i] for i in order
        ],
    }


@traceable(name="process_context", run_type="prompt")
def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"- ID: {id}, Rating: {rating}, Description: {chunk}\n"
    return formatted_context


@traceable(name="build_prompt", run_type="prompt")
def build_prompt(preprocessed_context, question):
    prompt_path = Path(__file__).parent / "prompts" / "retrieval_generation.yaml"
    template = prompt_template_config(
        yaml_path=prompt_path,
        prompt_key="retrieval_generation",
    )
    prompt = template.render(
        preprocessed_context=preprocessed_context, question=question
    )
    return prompt


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_model_name": "gpt-5.4-nano", "ls_provider": "openai"},
)
def generate_answer(prompt):
    client = instructor.from_provider(
        "openai/gpt-5.4-nano", mode=instructor.Mode.RESPONSES_TOOLS
    )
    # raw response is needed for cost calculation
    response, raw_response = client.create_with_completion(
        messages=[
            {"role": "system", "content": prompt},
        ],
        reasoning={"effort": "none"},
        response_model=RAGGenerationResponse,
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
    return response


@traceable(
    name="rag_pipeline",
)
def rag_pipeline(
    question, qdrant_client, top_k=5, hybrid=True, rerank=False, retrieve_k=20
):

    retrieved_context = retrieve_from_qdrant(
        question, qdrant_client, k=retrieve_k if rerank else top_k, hybrid=hybrid
    )

    if rerank:
        retrieved_context = rerank_data_openrouter(
            question, retrieved_context, top_k=top_k
        )

    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)

    final_answer = {
        "answer": answer.answer,
        "references": answer.references,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
    }

    return final_answer


def rag_pipeline_wraper(question, top_k=5):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result = rag_pipeline(question, qdrant_client, top_k)

    used_context = []

    for item in result.get("references", []):
        payload = qdrant_client.scroll(
            collection_name="Amazon-items-collection-01-hybrid-search",
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="parent_asin", match=MatchValue(value=item.id))
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
                    "description": item.description,
                }
            )

    return {"answer": result["answer"], "used_context": used_context}
