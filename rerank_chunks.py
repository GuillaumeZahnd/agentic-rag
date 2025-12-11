from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import Tuple

from timer import sync_timer


@sync_timer
def rerank_chunks(
        cross_encoder: CrossEncoder,
        query: str,
        chunks: list[Document],
        top_k_chunks: int) -> list[Tuple[Document, float]]:
    """
    Rerank and trim a list of Document based on contextual relevance with the query, using a cross-encoder.
    This operation has a higher precision than the bi-encoder, but is slower and has a lower recall.

    Args:
        cross_encoder: Instance of the cross-encoder model.
        query: User query string.
        chunks: List of candidate documents retrieved from the vector store.
        top_k_chunks: Number of the most relevant documents that must be selected.

    Returns:
        List of the most relevant chunks, sorted by decreasing relevance with the query, of length (top_k_chunks).
    """

    # Associate each chunk to the query to create (query, content) pairs
    query_content_pairs = [[query, chunk.page_content] for chunk in chunks]

    # Score each (query, content) pair based on contextual relevance
    scores = cross_encoder.predict(query_content_pairs)

    # Convert the scores to Python floats, to avoid unnecessary "np.float32(...)"
    scores = [float(s) for s in scores]

    # Create a list of tuples (chunk, score)
    reranked_chunks = list(zip(chunks, scores))

    # Sort the list by decreasing scores
    reranked_chunks = sorted(reranked_chunks, key=lambda x: x[1], reverse=True)

    # Trims the list to preserve only the most relevant chunks
    selected_chunks = [{"chunk": chunk, "score": score} for chunk, score in reranked_chunks[:top_k_chunks]]

    return selected_chunks

