import openai
import cohere
import time
from urllib.error import HTTPError, URLError
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
from qdrant_client.models import Prefetch, Document
from qdrant_client import models
from langchain_core.tools import tool
from urllib.request import Request, urlopen
import json, os
from qdrant_client.models import Filter, FieldCondition, MatchAny, FusionQuery

### Items metadata retrieval tool


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
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


@traceable(name="retrieve_items_data", run_type="retriever")
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
def rerank_data_openrouter(
    query, context, top_k=5, model="cohere/rerank-4-pro", max_retries=3
):
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


@traceable(name="format_retrieved_context", run_type="prompt")
def process_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


@tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """Search available products and return the top k matching inventory items.

    Expand the customer's question into 1-5 concise search statements and issue them
    in parallel in a single turn. Each statement covers one distinct product or
    attribute; no two may express the same intent. Use natural product-description
    language. If no brand or model is specified, search broadly rather than refusing.

        "Earphones for me and a waterproof speaker"
            -> "Personal earphones" | "Waterproof speaker"
        "A warm winter jacket for hiking"
            -> "Insulated winter jacket" | "Hiking outerwear for cold weather"

    Before calling, check what earlier calls in this conversation already returned.
    Search only for what is missing; results already retrieved remain valid and must
    not be fetched again.

    Args:
        query: A single search statement describing one product or attribute.
        top_k: Number of items to retrieve. Works best with 5 or more.

    Returns:
        A string of the top k available products, each prefixed with its ID and
        average rating."""

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    retrieved_context = retrieve_items_data(query, qdrant_client, k=20)

    retrieved_context = rerank_data_openrouter(query, retrieved_context, top_k=top_k)
    formatted_context = process_context(retrieved_context)

    return formatted_context


# def get_embedding(text, model="text-embedding-3-small"):
#     """因为 OpenAI Embeddings API 支持一次输入多条文本，所以返回的 data 始终是列表。当前只传入一条文本："""
#     response = openai.embeddings.create(input=text, model=model)

#     return response.data[0].embedding


def retrieve_prefiltered_reviews_data(query, parent_asins, k=5, qdrant_client=None):

    query_embedding = get_embedding(query)

    if qdrant_client is None:
        qdrant_client = QdrantClient(url="http://qdrant:6333")

    results = qdrant_client.query_points(
        collection_name="Amazon-reviews-collection-01",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin", match=MatchAny(any=parent_asins)
                        )
                    ]
                ),
                limit=20,
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["preprocessed_data"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }


def process_reviews_context(context):

    formatted_context = ""

    for id, chunk in zip(
        context["retrieved_context_ids"], context["retrieved_context"]
    ):
        formatted_context += f"- ID: {id}, user review: {chunk}\n"

    return formatted_context


@tool
def get_formatted_reviews_context(
    query: str, parent_asins: list[str], top_k: int = 5
) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.

    Args:
        query: The query to get the top k reviews for
        parent_asins: The item IDs used to prefilter reviews before retrieval
        top_k: The number of reviews to retrieve, this should be at least 20 if multipple items are prefiltered

    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    retrieved_context = retrieve_prefiltered_reviews_data(
        query, parent_asins, top_k, qdrant_client=qdrant_client
    )
    formatted_context = process_reviews_context(retrieved_context)

    return formatted_context
