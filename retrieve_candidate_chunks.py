from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from timer import timer


@timer
def retrieve_candidate_chunks(
        user_query: str, vector_store: VectorStore, top_k_chunks: int) -> list[Document]:
    """
    Compute the similarity between the user query and each chunks, based on vector distance, using a bi-encoder.
    This operation has a higher recall than the cross-encoder and is also faster, but has a lower precision.

    Args:
        user_query: User query string.
        vector_store: Vector store containing the pre-embedded documents chunks.
        top_k_chunks: Number of the most relevant documents that must be selected.

    Returns:
        List of the most relevant chunks, sorted by increasing distance from the query vector, of length (top_k_chunks).
    """

    retrieved_documents = vector_store.similarity_search(query=user_query, k=top_k_chunks)

    return retrieved_documents
