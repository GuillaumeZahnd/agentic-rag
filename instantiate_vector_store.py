from langchain_chroma import Chroma


def instantiate_vector_store(embeddings_model):

    collection_name = "collection_placeholder_name"
    persist_directory = "./chroma_langchain_db"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings_model,
        persist_directory=persist_directory)

    return vector_store
