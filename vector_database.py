from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import asyncio


class VectorDatabase:
    def __init__(self, embedding_model: Embeddings):
        super().__init__()

        collection_name = "collection_placeholder_name"
        persist_directory = "./chroma_langchain_db"

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory)


    async def a_retrieve_via_thread(self, query: str, k: int) -> list[Document]:
        """
        Chroma does not provide a built-in, native asynchronous retrieval method (e.g., a_retrieve or a_similarity_search).
        This custom function executes the synchronous method 'similarity_search' in a separate thread.
        """
        return await asyncio.to_thread(self.vector_store.similarity_search, query, k)
