import openai
from qdrant_client.models import (
    Prefetch,
    Document,
    FusionQuery,
    Filter,
    FieldCondition,
    MatchAny,
)
from qdrant_client import models
import cohere
from urllib.request import Request, urlopen
import json
import os
import time
from urllib.error import HTTPError, URLError

# we do not have tracing as it is mcp on server side


def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)

    return response.data[0].embedding


def retrieve_items_data(query, qdrant_client, k=5, hybrid=True):

    query_embedding = get_embedding(query)

    if hybrid:
        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            prefetch=[
                Prefetch(
                    query=query_embedding, using="text-embedding-3-small", limit=20
                ),
                Prefetch(
                    query=Document(text=query, model="qdrant/bm25"),
                    using="bm25",
                    limit=20,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(weights=[3, 1])),
            limit=k,
        )
    else:
        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            query=query_embedding,
            using="text-embedding-3-small",
            limit=k,
        )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["preprocessed_description"])
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings,
    }


# def rerank_data(query, context, top_k=5):

#     cohere_client = cohere.ClientV2()

#     response = cohere_client.rerank(
#         model="rerank-v4.0-pro",
#         query=query,
#         documents=context["retrieved_context"],
#         top_n=top_k,
#     )

#     order = [result.index for result in response.results]

#     return {
#         "retrieved_context_ids": [context["retrieved_context_ids"][i] for i in order],
#         "retrieved_context": [context["retrieved_context"][i] for i in order],
#         "similarity_scores": [context["similarity_scores"][i] for i in order],
#         "retrieved_context_ratings": [
#             context["retrieved_context_ratings"][i] for i in order
#         ],
#     }


def rerank_data(query, context, top_k=5, model="cohere/rerank-4-pro", max_retries=3):
    request_body = {
        "model": model,
        "query": query,
        "documents": context["retrieved_context"],
        "top_n": top_k,
    }
    retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504, 524, 529}
    last_error = None

    for attempt in range(max_retries + 1):
        request = Request(
            "https://openrouter.ai/api/v1/rerank",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                response_body = json.loads(response.read().decode("utf-8"))

            if "results" in response_body:
                break

            error = response_body.get("error", {})
            status = error.get("code") if isinstance(error, dict) else None
            try:
                status = int(status) if status is not None else None
            except (TypeError, ValueError):
                status = None

            last_error = RuntimeError(
                f"OpenRouter rerank returned no results: {response_body}"
            )
            if status is not None and status not in retryable_statuses:
                raise last_error

        except HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"OpenRouter rerank HTTP {exc.code}: {response_text}"
            )
            if exc.code not in retryable_statuses:
                raise last_error from exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc

        if attempt == max_retries:
            raise RuntimeError(
                f"OpenRouter rerank failed after {max_retries + 1} attempts"
            ) from last_error

        time.sleep(2**attempt)

    order = [result["index"] for result in response_body["results"]]
    return {
        "retrieved_context_ids": [context["retrieved_context_ids"][i] for i in order],
        "retrieved_context": [context["retrieved_context"][i] for i in order],
        "similarity_scores": [context["similarity_scores"][i] for i in order],
        "retrieved_context_ratings": [
            context["retrieved_context_ratings"][i] for i in order
        ],
    }


def process_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context
