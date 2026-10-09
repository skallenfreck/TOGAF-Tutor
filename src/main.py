from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag.documents import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_documents,
    load_directory,
)
from src.rag.embeddings import DEFAULT_EMBEDDING_MODEL, get_local_embedder
from src.rag.generation import DEFAULT_GROQ_MODEL
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import ChromaVectorStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_documents"
INDEX_DIR = ROOT / "data" / "chroma"


def _client_from_environment():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Falta GROQ_API_KEY. Configura una clave de Groq para habilitar la generación."
        )
    from groq import Groq

    return Groq(api_key=api_key)


def _ensure_index() -> ChromaVectorStore:
    store = ChromaVectorStore(path=INDEX_DIR)
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    metadata = store.collection.metadata or {}
    if (
        store.count() > 0
        and metadata.get("embedding_model") == model_name
        and metadata.get("chunk_size") == DEFAULT_CHUNK_SIZE
        and metadata.get("chunk_overlap") == DEFAULT_CHUNK_OVERLAP
    ):
        return store

    documents = load_directory(SOURCE_DIR)
    chunks = chunk_documents(
        documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
    )
    embedder = get_local_embedder(model_name)
    store.replace_index(
        chunks,
        embedder,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
    )
    print(f"Índice creado con {len(chunks)} fragmentos de {len(documents)} páginas/documentos.")
    return store


def main() -> None:
    load_dotenv(ROOT / ".env")
    print("=================================")
    print("      TUTOR TOGAF - RAG")
    print("=================================")
    try:
        client = _client_from_environment()
        store = _ensure_index()
        embedder = get_local_embedder(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Configuración pendiente: {exc}")
        print("Revisa el README para agregar los documentos y configurar la API.")
        return

    pipeline = RAGPipeline(
        store,
        embedder,
        min_similarity=float(os.getenv("MIN_RETRIEVAL_SIMILARITY", "0.55")),
        model_name=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
    )
    history: list[dict[str, str]] = []
    print("Escribe 'salir' para terminar.\n")
    while True:
        question = input("Tú: ").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            break
        if not question:
            continue
        try:
            answer, sources = pipeline.answer(question, history, client=client)
        except Exception as exc:
            print(f"No se pudo generar la respuesta: {exc}\n")
            continue

        print("Tutor TOGAF:")
        print(json.dumps(answer.model_dump(), ensure_ascii=False, indent=2))
        if sources:
            print("Fuentes recuperadas:")
            for item in sources:
                location = str(item.metadata.get("source", "Documento"))
                if item.metadata.get("page") is not None:
                    location += f", página {item.metadata['page']}"
                print(f"- {location} (similitud {item.similarity:.3f})")
        print()
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer.respuesta},
            ]
        )


if __name__ == "__main__":
    main()
