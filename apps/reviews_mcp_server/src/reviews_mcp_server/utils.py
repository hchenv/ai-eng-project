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
from qdrant_client import QdrantClient

# we do not have tracing as it is mcp on server side


def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)

    return response.data[0].embedding


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
