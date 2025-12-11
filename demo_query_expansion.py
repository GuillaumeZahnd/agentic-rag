import os
from langchain_core.documents import Document

from query_expansion import QueryExpansion


if __name__ == "__main__":

    user_query = "what is a permanent?"
    nb_variants = 3
    temperature = 0.1

    with open(os.path.join("data", "mtg.txt"), "r") as fid:
        context = fid.read()

    document = Document(page_content=context)

    chunks = []
    chunks.append(document)

    query_expansion = QueryExpansion(nb_variants=nb_variants, temperature=temperature)

    _, log_message = query_expansion.expand_query(query=user_query, chunks=chunks)

    print(log_message)
