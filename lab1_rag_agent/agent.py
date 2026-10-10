import chromadb
from sentence_transformers import SentenceTransformer


DB_PATH = "vector_db"
COLLECTION_NAME = "warehouse_documents"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

print("Подключение к ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(f"Количество записей в базе: {collection.count()}")

print("Загрузка модели embeddings...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Модель embeddings загружена.")

def search_docs(query, filters=None, n_results=5):
    """
    Выполняет семантический поиск по корпоративным документам.

    query:
        Поисковый запрос пользователя.

    filters:
        Фильтры по метаданным.
        Например:
        {"doc_type": "xlsx"}

    n_results:
        Количество результатов.
    """

    print("\n[TOOL] search_docs")
    print(f"Запрос: {query}")

    if filters:
        print(f"Фильтры: {filters}")

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    search_params = {
        "query_embeddings": query_embedding,
        "n_results": n_results
    }

    if filters:
        search_params["where"] = filters

    results = collection.query(
        **search_params
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances
    }

def get_summary(doc_type):
    """
    Получает информацию из документов
    определённого типа.
    """

    print("\n[TOOL] get_summary")
    print(f"Тип документа: {doc_type}")

    results = collection.get(
        where={
            "doc_type": doc_type
        }
    )

    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    return {
        "documents": documents,
        "metadatas": metadatas
    }

def detect_document_type(query):
    """
    Определяет тип документа на основании вопроса пользователя.
    """

    query_lower = query.lower()

    # Excel
    excel_words = [
        "excel",
        "эксель",
        "таблиц",
        "таблица",
        "каталог",
        "товар",
        "товары"
    ]

    for word in excel_words:
        if word in query_lower:
            return "xlsx"

    csv_words = [
        "инвентар",
        "остатк",
        "недостач",
        "склад"
    ]

    for word in csv_words:
        if word in query_lower:
            return "csv"

    supplier_words = [
        "поставщик",
        "поставщики",
        "поставщикам",
        "supplier"
    ]

    for word in supplier_words:
        if word in query_lower:
            return "docx"

    rules_words = [
        "правил",
        "регламент",
        "регламенте",
        "правилах"
    ]

    for word in rules_words:
        if word in query_lower:
            return "txt"

    pdf_words = [
        "pdf",
        "складск",
        "безопасност"
    ]

    for word in pdf_words:
        if word in query_lower:
            return "pdf"

    return None

def choose_tool(query):
    """
    Определяет инструмент для выполнения запроса.
    """

    query_lower = query.lower()

    summary_words = [
        "сводка",
        "сводку",
        "итог",
        "итоги",
        "обзор",
        "кратко расскажи",
        "что содержится",
        "что есть в документе",
        "покажи данные",
        "покажи информацию"
    ]

    for word in summary_words:
        if word in query_lower:
            return "summary"

    return "search"

def print_search_results(results):
    """
    Выводит результаты семантического поиска.
    """

    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    distances = results.get(
        "distances",
        []
    )

    print()
    print("=" * 70)
    print("РЕЗУЛЬТАТ ПОИСКА")
    print("=" * 70)

    if not documents:
        print("Информация не найдена.")
        return

    for i, document in enumerate(documents):

        print()
        print(f"[{i + 1}] Результат")

        if i < len(metadatas):

            metadata = metadatas[i]

            print(
                f"Источник: "
                f"{metadata.get('source', 'не указан')}"
            )

            print(
                f"Тип документа: "
                f"{metadata.get('doc_type', 'не указан')}"
            )

            print(
                f"Дата документа: "
                f"{metadata.get('date', 'не указана')}"
            )

        if i < len(distances):

            print(
                f"Расстояние: "
                f"{distances[i]:.4f}"
            )

        print("Содержимое:")
        print(document)

        print("-" * 70)

def print_summary_results(results):
    """
    Выводит результаты получения сводки.
    """

    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    print()
    print("=" * 70)
    print("СВОДКА ПО ДОКУМЕНТУ")
    print("=" * 70)

    if not documents:
        print("Документы указанного типа не найдены.")
        return

    for i, document in enumerate(documents):

        print()
        print(f"[{i + 1}]")

        if i < len(metadatas):

            metadata = metadatas[i]

            print(
                f"Источник: "
                f"{metadata.get('source', 'не указан')}"
            )

            print(
                f"Тип: "
                f"{metadata.get('doc_type', 'не указан')}"
            )

            print(
                f"Дата: "
                f"{metadata.get('date', 'не указана')}"
            )

        print()
        print(document)

        print("-" * 70)

def run_agent():

    print()
    print("=" * 70)
    print("ЛОКАЛЬНЫЙ RAG-АГЕНТ")
    print("=" * 70)

    print()
    print("Доступные инструменты:")
    print("1. search_docs(query, filters)")
    print("2. get_summary(doc_type)")

    print()
    print("Агент умеет:")
    print("- выполнять семантический поиск")
    print("- фильтровать документы по типу")
    print("- получать сводку документов")
    print("- работать с PDF, DOCX, XLSX, CSV и TXT")

    print()
    print("Введите 'exit' для выхода.")

    while True:

        query = input("\nВаш вопрос: ").strip()

        if query.lower() == "exit":

            print()
            print("Работа агента завершена.")
            break

        if not query:
            continue

        selected_tool = choose_tool(query)

        print()
        print(
            f"[AGENT] Выбран инструмент: "
            f"{selected_tool}"
        )

        if selected_tool == "search":

            doc_type = detect_document_type(query)

            filters = None

            if doc_type:

                filters = {
                    "doc_type": doc_type
                }

                print(
                    f"[AGENT] Определён тип документа: "
                    f"{doc_type}"
                )

            results = search_docs(
                query=query,
                filters=filters,
                n_results=5
            )

            print_search_results(results)

        elif selected_tool == "summary":

            doc_type = detect_document_type(query)

            if doc_type:

                print(
                    f"[AGENT] Определён тип документа: "
                    f"{doc_type}"
                )

            else:

                print(
                    "[AGENT] Тип документа "
                    "не удалось определить."
                )

                print(
                    "[AGENT] Будет использован "
                    "семантический поиск."
                )

                results = search_docs(
                    query=query,
                    filters=None,
                    n_results=5
                )

                print_search_results(results)

                continue

            results = get_summary(
                doc_type
            )

            print_summary_results(results)

if __name__ == "__main__":
    run_agent()