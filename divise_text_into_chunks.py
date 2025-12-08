from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def divise_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True)
    docs = splitter.split_documents([Document(page_content=text)])
    return docs
