import os
from langchain_core.documents import Document
from typing import Tuple
from langchain.messages import AIMessage


def log_llm_query_answering(
    query: str,
    answer: str,
    chunks: list[Tuple[Document, float]],
    prompt: str,
    raw_answer: AIMessage) -> str:

    log_message = []
    log_message.append("# User query:\n")
    log_message.append(query)
    log_message.append("-"*64)
    log_message.append("# Answer:\n")
    log_message.append(answer)
    log_message.append("-"*64)
    log_message.append("# Reranked chunks:\n")
    for i in range(len(chunks)):
        log_message.append("[{}] Score: {}".format(i, chunks[i]["score"]))
        log_message.append("{}\n\n".format(chunks[i]["chunk"].page_content))
    log_message.append("-"*64)
    log_message.append("# Prompt:\n")
    log_message.append(prompt)
    log_message.append("-"*64)
    log_message.append("# Raw answer:\n")
    log_message.append(str(raw_answer))

    log_message = "\n".join(log_message)

    path_to_logs = "logs"
    if not os.path.exists(path_to_logs):
        os.makedirs(path_to_logs)

    with open(os.path.join(path_to_logs, "log_llm_query_answering.txt"), "w") as fid:
        fid.write(log_message)

    return log_message
