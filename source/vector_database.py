from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import asyncio
import hashlib

from source.extract_text_from_pdf import extract_text_from_pdf
from source.divise_text_into_chunks import divise_text_into_chunks
from source.log_vector_store import log_vector_store
from source.timer import sync_timer


class VectorDatabase:
    def __init__(self, embedding_model: Embeddings):
        super().__init__()

        collection_name = "collection_placeholder_name"
        persist_directory = "./chroma_langchain_db"

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory)

        self.chunk_size = 500
        self.chunk_overlap = 100


    @sync_timer
    def populate_vector_store(self, pdf_url: str) -> None:

        text = extract_text_from_pdf(url=pdf_url)

        chunks = divise_text_into_chunks(text=text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        collection = self.vector_store.get()
        existing_ids = set(collection["ids"])

        chunks_to_add: List[Document] = []
        ids_to_add: List[str] = []

        for chunk in chunks:
            chunk_id = self._get_chunk_id(chunk)

            if chunk_id not in existing_ids:
                chunks_to_add.append(chunk)
                ids_to_add.append(chunk_id)

        nb_chunks_to_add = len(chunks_to_add)

        if nb_chunks_to_add > 0:
            _ = self.vector_store.add_documents(documents=chunks_to_add, ids=ids_to_add)

        log_vector_store(
            chunks=chunks,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            nb_chunks_to_add=nb_chunks_to_add)


    async def a_retrieve_via_thread(self, query: str, k: int) -> list[Document]:
        """
        Chroma does not provide a built-in, native asynchronous retrieval method (e.g., a_retrieve or a_similarity_search).
        This custom function executes the synchronous method 'similarity_search' in a separate thread.
        """
        return await asyncio.to_thread(self.vector_store.similarity_search, query, k)


    def _get_chunk_id(self, chunk: Document) -> str:
        content = chunk.page_content.encode("utf-8")
        metadata = str(chunk.metadata).encode("utf-8")
        return hashlib.sha256(content + metadata).hexdigest()
