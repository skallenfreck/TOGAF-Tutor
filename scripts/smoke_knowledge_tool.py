"""Demostración de buscar_en_base sin llamada a un modelo generativo."""
from __future__ import annotations

import json
from pathlib import Path

from src.agent.tools import BuscarEnBaseTool
from src.rag.documents import chunk_documents, load_directory
from src.rag.embeddings import DEFAULT_EMBEDDING_MODEL, get_local_embedder
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import ChromaVectorStore


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = load_directory(root / "data" / "source_documents")
    chunks = chunk_documents(sources)
    embedder = get_local_embedder(DEFAULT_EMBEDDING_MODEL)
    store = ChromaVectorStore(path=None, collection_name="tool_smoke_test")
    store.replace_index(chunks, embedder)
    pipeline = RAGPipeline(store, embedder, min_similarity=0.55)
    tool = BuscarEnBaseTool(pipeline)

    result = tool.invoke({"consulta": "¿Cuál es el objetivo de la fase de Visión de Arquitectura?"})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
