from fastapi import FastAPI
from pydantic import BaseModel

from agent import (
    search_docs,
    get_summary,
    detect_document_type,
    choose_tool
)

app = FastAPI(
    title="Warehouse RAG Agent API",
    description="API для работы с локальным RAG-агентом",
    version="1.0"
)

class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Warehouse RAG Agent API работает",
        "endpoints": [
            "GET /",
            "POST /ask",
            "GET /docs"
        ]
    }


def format_results(results):
    formatted = []

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    distances = results.get("distances", [])

    if documents and isinstance(documents[0], list):
        documents = documents[0]

    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]

    if distances and isinstance(distances[0], list):
        distances = distances[0]

    for i, document in enumerate(documents):

        metadata = {}

        if i < len(metadatas):
            metadata = metadatas[i]

        distance = None

        if i < len(distances):
            distance = distances[i]

        formatted.append({
            "source": metadata.get("source"),
            "doc_type": metadata.get("doc_type"),
            "date": metadata.get("date"),
            "distance": distance,
            "content": document
        })

    return formatted

@app.post("/ask")
def ask(request: QueryRequest):

    query = request.query.strip()

    if not query:

        return {
            "status": "error",
            "message": "Запрос не может быть пустым"
        }

    tool = choose_tool(query)

    document_type = detect_document_type(query)

    if tool == "search":

        filters = None

        if document_type:
            filters = {
                "doc_type": document_type
            }

        results = search_docs(
            query=query,
            filters=filters,
            n_results=5
        )

        formatted_results = format_results(results)

        return {
            "status": "success",
            "query": query,
            "tool": "search",
            "document_type": document_type,
            "results": formatted_results
        }

    elif tool == "summary":

        if not document_type:

            return {
                "status": "error",
                "query": query,
                "tool": "summary",
                "message": "Не удалось определить тип документа"
            }

        results = get_summary(document_type)

        formatted_results = format_results(results)

        return {
            "status": "success",
            "query": query,
            "tool": "summary",
            "document_type": document_type,
            "results": formatted_results
        }


    return {
        "status": "error",
        "query": query,
        "message": f"Неизвестный инструмент: {tool}"
    }