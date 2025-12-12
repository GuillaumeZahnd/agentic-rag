import os
from langchain_core.documents import Document


def log_vector_store(
    chunks: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    nb_chunks_to_add: int) -> str:

    log_message = []
    log_message.append("# Number of chunks :\n")
    log_message.append(str(len(chunks)))
    log_message.append("-"*64)
    log_message.append("# Number of new chunks that were added:\n")
    log_message.append(str(nb_chunks_to_add))
    log_message.append("-"*64)
    log_message.append("# Chunk size :\n")
    log_message.append(str(chunk_size))
    log_message.append("-"*64)
    log_message.append("# Chunk overlap :\n")
    log_message.append(str(chunk_overlap))
    log_message.append("-"*64)
    log_message.append("# Chunks:\n")
    for index, chunk in enumerate(chunks):
        log_message.append("[{}] {}\n\n".format(index, chunk))

    log_message = "\n".join(log_message)

    path_to_logs = "logs"
    if not os.path.exists(path_to_logs):
        os.makedirs(path_to_logs)

    with open(os.path.join(path_to_logs, "log_vector_store.txt"), "w") as fid:
        fid.write(log_message)

    return log_message
