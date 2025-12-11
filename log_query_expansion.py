import os
from langchain_core.documents import Document


def log_query_expansion(
        query: str,
        chunks: list[Document],
        alternative_queries: list[str],
        alternative_queries_raw: str,
        prompt: str) -> str:

    log_message = []
    log_message.append("# User query:\n")
    log_message.append(query)
    log_message.append("-"*64)
    log_message.append("# Retrieved chunks:\n")
    for index, chunk in enumerate(chunks):
        log_message.append("[{}] {}\n\n".format(index, chunk.page_content))
    log_message.append("-"*64)
    log_message.append("# Alternative queries:\n")
    for index, query in enumerate(alternative_queries):
        log_message.append("[{}] {}".format(index, query))
    log_message.append("-"*64)
    log_message.append("# Alternative queries (raw output verification):\n")
    log_message.append(alternative_queries_raw)
    log_message.append("-"*64)
    log_message.append("# Prompt:\n")
    log_message.append(prompt)

    log_message = "\n".join(log_message)

    path_to_logs = "logs"
    if not os.path.exists(path_to_logs):
        os.makedirs(path_to_logs)

    with open(os.path.join(path_to_logs, "log_query_expansion.txt"), "w") as fid:
        fid.write(log_message)

    return log_message
