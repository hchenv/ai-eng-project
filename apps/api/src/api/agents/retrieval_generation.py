import openai
from langsmith import traceable, get_current_run_tree 
# for cost calculation
from qdrant_client import QdrantClient
from api.core.config import config

qdrant_client = QdrantClient(url = "http://qdrant:6333") # on docker image it called qdrant as defined

@traceable(name="get_embedding", run_type = "embedding",metadata = {"ls_model_name": "text-embedding-3-small", "ls_provider": "openai"})
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }
    return response.data[0].embedding

@traceable(name="retrieve_from_qdrant", run_type = "retriever")
def retrieve_from_qdrant(query,qdrant_client, k=5):
    response = qdrant_client.query_points(
        collection_name="amazon-electronics-items-collection-01",
        query=get_embedding(query),
        limit=k
    )
    retrieved_context_ids = []
    retrived_context = []
    similarity_scores = []
    retrived_context_ratings = []
    for point in response.points:
        retrieved_context_ids.append(point.payload['parent_asin'])
        retrived_context.append(point.payload['preprocessed_description'])
        similarity_scores.append(point.score)
        retrived_context_ratings.append(point.payload['average_rating'])
        
    return {"retrieved_context_ids": retrieved_context_ids,
            "retrived_context": retrived_context,
            "similarity_scores": similarity_scores,
            "retrived_context_ratings": retrived_context_ratings}

@traceable(name="process_context", run_type = "prompt")
def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(context['retrieved_context_ids'], context['retrived_context'], context['retrived_context_ratings']):
        formatted_context += f"- ID: {id}, Rating: {rating}, Description: {chunk}\n"
    return formatted_context


@traceable(name="build_prompt", run_type = "prompt")
def build_prompt(preprocessed_context, question):
    prompt = f"""
    You are a helpful shopping assistant that can answer questions about the product in stock.

    You will be given a question and a list of context.

    Instructions:
    - Answer the question based on the provided context only.
    - Never use word context and refer to it as the available product.
    - Do not use markdown formatting.

    Here is the context:
    {preprocessed_context}
    Here is the question:
    {question}
    """
    return prompt


@traceable(name="generate_answer", run_type = "llm", metadata = {"ls_model_name": "gpt-5.4-nano", "ls_provider": "openai"})
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": prompt},
        ],
        reasoning_effort="none"
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    return response.choices[0].message.content


@traceable(name="rag_pipeline")
def rag_pipeline(question, qdrant_client, k=5):
    retrieved_context = retrieve_from_qdrant(question, qdrant_client, k=k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)
    final_answer = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieved_context['retrieved_context_ids'],
        "retrieved_context": retrieved_context['retrived_context']
    }
    return final_answer
