import openai
import cohere
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
from qdrant_client.models import Prefetch, Document
from qdrant_client import models
from langchain_core.tools import tool
from urllib.request import Request, urlopen
import json, os

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


@traceable(name="retrieve_data", run_type="retriever")
def retrieve_data(query, qdrant_client, k=5, hybrid=True):

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
    """Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    retrieved_context = retrieve_data(query, qdrant_client, k=20)

    retrieved_context = rerank_data_openrouter(query, retrieved_context, top_k=top_k)
    formatted_context = process_context(retrieved_context)

    return formatted_context
