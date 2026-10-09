from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.rag.documents import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DocumentChunk,
)
from src.rag.embeddings import LocalEmbedder


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def similarity(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.distance))


class ChromaVectorStore:
    """Chroma index with manually generated local embeddings."""

    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str = "togaf_knowledge",
    ) -> None:
        import chromadb

        self.collection_name = collection_name
        if path is None:
            self.client = chromadb.EphemeralClient()
        else:
            store_path = Path(path)
            store_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(store_path))
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        existing = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        return existing

    def replace_index(
        self,
        chunks: list[DocumentChunk],
        embedder: LocalEmbedder,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if not chunks:
            raise ValueError("No hay fragmentos de texto para indexar.")

        # Finish local embedding before replacing a persisted collection. A failed
        # model load/download must not erase the last usable index.
        batch_size = 64
        batches = [chunks[start:start + batch_size] for start in range(0, len(chunks), batch_size)]
        embeddings = [
            vector
            for batch in batches
            for vector in embedder.embed_documents([chunk.text for chunk in batch])
        ]
        if len(embeddings) != len(chunks):
            raise ValueError("El modelo de embeddings no devolvió un vector por fragmento.")

        existing_names = {
            getattr(collection, "name", collection)
            for collection in self.client.list_collections()
        }
        if self.collection_name in existing_names:
            self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedder.model_name,
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
            },
            embedding_function=None,
        )

        offset = 0
        for batch in batches:
            batch_embeddings = embeddings[offset:offset + len(batch)]
            self.collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
                embeddings=batch_embeddings,
            )
            offset += len(batch)

    def search(
        self, query: str, embedder: LocalEmbedder, top_k: int = 5
    ) -> list[RetrievedChunk]:
        count = self.collection.count()
        if count == 0:
            return []
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if not stored_model:
            raise ValueError(
                "Este índice no registra el modelo de embeddings con el que fue creado. "
                "Vuelve a crearlo con «python -m scripts.build_index»."
            )
        if stored_model and stored_model != embedder.model_name:
            raise ValueError(
                f"El índice usa embeddings de «{stored_model}», pero la consulta usa "
                f"«{embedder.model_name}». Vuelve a crear el índice con el modelo configurado."
            )
        results = self.collection.query(
            query_embeddings=[embedder.embed_query(query)],
            n_results=min(max(1, top_k), count),
            include=["documents", "metadatas", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        return [
            RetrievedChunk(
                text=document or "",
                metadata=metadata or {},
                distance=float(distance),
            )
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]

    def count(self) -> int:
        return int(self.collection.count())
