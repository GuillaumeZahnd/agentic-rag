from langchain_huggingface import HuggingFaceEmbeddings


def instantiate_embedding_model():

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True, "batch_size": 32}

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs)

    return embedding_model
