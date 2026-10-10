from pathlib import Path
import pandas as pd

from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = Path("documents")


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def load_docx(file_path):
    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    return "\n".join(paragraphs)


def load_csv(file_path):
    df = pd.read_csv(file_path)

    return df.to_string(index=False)


def load_xlsx(file_path):
    df = pd.read_excel(file_path)

    return df.to_string(index=False)


def load_document(file_path):
    extension = file_path.suffix.lower()

    if extension == ".txt":
        return load_txt(file_path)

    elif extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    elif extension == ".csv":
        return load_csv(file_path)

    elif extension == ".xlsx":
        return load_xlsx(file_path)

    else:
        raise ValueError(f"Неподдерживаемый формат: {extension}")


def get_document_date(file_name):
    dates = {
        "warehouse_rules.pdf": "2026-09-15",
        "product_catalog.xlsx": "2026-09-20",
        "suppliers.docx": "2026-09-20",
        "inventory.csv": "2026-09-30",
        "warehouse_regulations.txt": "2026-09-01",
    }

    return dates.get(file_name, "unknown")


def load_all_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.iterdir():

        if not file_path.is_file():
            continue

        print(f"Загрузка: {file_path.name}")

        text = load_document(file_path)

        metadata = {
            "source": file_path.name,
            "doc_type": file_path.suffix.lower().replace(".", ""),
            "date": get_document_date(file_path.name),
        }

        documents.append({
            "text": text,
            "metadata": metadata
        })

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for document in documents:

        text_chunks = splitter.split_text(document["text"])

        for chunk in text_chunks:

            chunks.append({
                "text": chunk,
                "metadata": document["metadata"]
            })

    return chunks


if __name__ == "__main__":

    documents = load_all_documents()

    print()
    print(f"Загружено документов: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Получено чанков: {len(chunks)}")

    print()
    print("ПРИМЕР ЧАНКА:")
    print(chunks[0]["text"])

    print()
    print("МЕТАДАННЫЕ:")
    print(chunks[0]["metadata"])
