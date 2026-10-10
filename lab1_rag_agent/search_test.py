import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="vector_db")

collection = client.get_collection(
    name="warehouse_documents"
)

print("Загрузка модели...")
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Модель загружена.")

query = "Какие товары имеют недостачу?"


query_embedding = model.encode(query).tolist()


results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print()
print("ВОПРОС:")
print(query)


for i in range(len(results["documents"][0])):

    print()
    print(f"РЕЗУЛЬТАТ №{i + 1}")

    print("Текст:")
    print(results["documents"][0][i])

    print()
    print("Метаданные:")
    print(results["metadatas"][0][i])

    print()
    print("Расстояние:")
    print(results["distances"][0][i])
