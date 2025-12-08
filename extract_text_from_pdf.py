import fitz


def extract_text_from_pdf(path_to_pdf: str) -> str:
    text = []
    with fitz.open(path_to_pdf) as document:
        for page in document:
            text.append(page.get_text("text"))
    text = "\n".join(text)
    return text
