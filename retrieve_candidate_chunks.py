from langchain_core.documents import Document
from vector_database import VectorDatabase
from typing import Coroutine
import asyncio

from timer import sync_timer, async_timer


async def retrieve_candidate_chunks(
    query: str | list[str], vector_database: VectorDatabase, top_k_chunks: int) -> list[Document]:
    """
    Compute the similarity between the user query and each chunks, based on vector distance, using a bi-encoder.
    This operation has a higher recall than the cross-encoder and is also faster, but has a lower precision.

    Args:
        query: Either a string for single query, or a list of strings for multi-query.
        vector_store: Vector store containing the pre-embedded documents chunks.
        top_k_chunks: Number of the most relevant documents that must be selected.

    Returns:
        List of the most relevant chunks, sorted by increasing distance from the query vector, of length (top_k_chunks).
    """

    if isinstance(query, str):
        return single_query_retrival(
                query=query,
                vector_database=vector_database,
                top_k_chunks=top_k_chunks)

    elif isinstance(query, list):
        return await multi_query_retrieval(
                queries=query,
                vector_database=vector_database,
                top_k_chunks=top_k_chunks)

    else:
        raise TypeError(
                "Input 'query' must be either of type str or list[str], but {} was received instead.".format(type(query).__name__))


@sync_timer
def single_query_retrival(
    query: str, vector_database: VectorDatabase, top_k_chunks: int) -> list[Document]:

    retrieved_documents = vector_database.vector_store.similarity_search(query=query, k=top_k_chunks)

    return retrieved_documents


@async_timer
async def multi_query_retrieval(
    queries: list[str], vector_database: VectorDatabase, top_k_chunks: int) -> list[Document]:

    # TODO fine-tune
    k = 1 + top_k_chunks // len(queries)

    tasks: list[Coroutine] = [vector_database.a_retrieve_via_thread(query, k) for query in queries]
    list_of_results_from_sequential_searches = await asyncio.gather(*tasks)
    all_retrieved_chunks = [chunk for chunk_list in list_of_results_from_sequential_searches for chunk in chunk_list]

    # Enforce uniqueness
    unique_chunks = {}
    for chunk in all_retrieved_chunks:
        chunk_id = chunk.metadata.get("id") or chunk.page_content
        unique_chunks[chunk_id] = chunk

    retrieved_chunks = list(unique_chunks.values())


    return retrieved_chunks
