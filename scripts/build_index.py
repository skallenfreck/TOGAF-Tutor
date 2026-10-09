from __future__ import annotations

import argparse
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
from src.rag.vector_store import ChromaVectorStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae, fragmenta y vectoriza el corpus TOGAF.")
    parser.add_argument("--documents", default=str(ROOT / "data" / "source_documents"))
    parser.add_argument("--index", default=str(ROOT / "data" / "chroma"))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    documents = load_directory(args.documents)
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    embedder = get_local_embedder(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    store = ChromaVectorStore(path=args.index)
    store.replace_index(
        chunks,
        embedder,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Documentos/páginas procesados: {len(documents)}")
    print(f"Fragmentos indexados: {store.count()}")
    print(f"Tamaño de fragmento: {args.chunk_size} caracteres; solapamiento: {args.overlap}.")
    print(f"Índice local persistido en: {Path(args.index).resolve()}")


if __name__ == "__main__":
    main()
