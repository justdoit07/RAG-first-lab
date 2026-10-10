import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_collection(
    name="warehouse_documents"
)

print("Загрузка модели embeddings...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Модель загружена.")

def search_documents(query, n_results=3):
    """
    Ищет наиболее подходящие фрагменты
    корпоративных документов.
    """

    query_embedding = model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas, distances


def build_context(documents, metadatas):
    """
    Объединяет найденные фрагменты
    в единый контекст для будущей LLM.
    """

    context_parts = []

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1
    ):

        source = metadata.get(
            "source",
            "неизвестный источник"
        )

        context_parts.append(
            f"""
Фрагмент №{i}
Источник: {source}

{document}
"""
        )

    return "\n".join(context_parts)

def generate_answer(query, documents, metadatas):
    """
    Пока используем простой механизм формирования ответа.
    На следующем этапе подключим LLM.
    """

    context = build_context(
        documents,
        metadatas
    )

    print()
    print("=" * 70)
    print("ВОПРОС")
    print("=" * 70)
    print(query)

    print()
    print("=" * 70)
    print("КОНТЕКСТ RAG")
    print("=" * 70)
    print(context)

    print()
    print("=" * 70)
    print("ИСТОЧНИКИ")
    print("=" * 70)

    unique_sources = []

    for metadata in metadatas:

        source = metadata.get(
            "source",
            "неизвестный источник"
        )

        if source not in unique_sources:
            unique_sources.append(source)

    for source in unique_sources:
        print(f"- {source}")

if __name__ == "__main__":

    query = input(
        "\nВведите вопрос: "
    )

    documents, metadatas, distances = search_documents(
        query,
        n_results=3
    )

    generate_answer(
        query,
        documents,
        metadatas
    )