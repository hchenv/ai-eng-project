from api.agents.retrieval_generation import rag_pipeline

from qdrant_client import QdrantClient
from langsmith import Client

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy

ls_client = Client()
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-5.4-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

qdrant_client = QdrantClient(url="http://localhost:6333")
# do not use async as it works better with langsmith?

def ragas_context_precision_id_based(run, example):

    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"]
    )

    scorer = IDBasedContextPrecision()

    return scorer.single_turn_score(sample)

def ragas_context_recall_id_based(run, example):

    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"]
    )
    # run.outputs becasue we run in langsmith? run is the trace from langsmith

    scorer = IDBasedContextRecall()

    return scorer.single_turn_score(sample)

def ragas_faithfulness(run, example):

    sample = SingleTurnSample(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"]
    )

    scorer = Faithfulness(llm=ragas_llm)

    return scorer.single_turn_score(sample)

def ragas_relevancy(run, example):

    sample = SingleTurnSample(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"]
    )

    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    return scorer.single_turn_score(sample)

## scan the dataset row by row and run the pipeline
rresults = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_context_precision_id_based,
        ragas_context_recall_id_based,
        ragas_faithfulness,
        ragas_relevancy
    ],
    experiment_prefix="retriever"
)