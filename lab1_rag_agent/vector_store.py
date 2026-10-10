import chromadb

from sentence_transformers import SentenceTransformer

from document_loader import load_all_documents, split_documents

documents = load_all_documents()

print(f"Загружено документов: {len(documents)}")


chunks = split_documents(documents)

print(f"Получено чанков: {len(chunks)}")


print("Загрузка модели embeddings...")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Модель загружена.")


client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_or_create_collection(
    name="warehouse_documents"
)

texts = []
metadatas = []
ids = []


for index, chunk in enumerate(chunks):

    texts.append(chunk["text"])

    metadatas.append(chunk["metadata"])

    ids.append(f"chunk_{index}")


print("Создание embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)


collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


print()
print("Данные успешно сохранены в ChromaDB.")
print(f"Количество записей: {collection.count()}")