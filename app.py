from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag.documents import chunk_documents, load_directory
from src.rag.embeddings import DEFAULT_EMBEDDING_MODEL, get_local_embedder
from src.rag.generation import DEFAULT_GROQ_MODEL
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import ChromaVectorStore

load_dotenv()
st.set_page_config(page_title="Tutor TOGAF", page_icon="📚", layout="centered")


@st.cache_resource(show_spinner="Cargando el modelo local de embeddings…")
def cached_embedder(model_name: str):
    return get_local_embedder(model_name)


@st.cache_resource(show_spinner=False)
def cached_groq_client(api_key: str):
    from groq import Groq

    return Groq(api_key=api_key)


def groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return ""


@st.cache_resource(show_spinner="Preparando el corpus TOGAF incluido…")
def load_builtin_index(model_name: str):
    """Build and cache the bundled TOGAF index once per Streamlit process."""
    corpus_dir = Path(__file__).resolve().parent / "data" / "source_documents"
    source_documents = load_directory(corpus_dir)
    chunks = chunk_documents(source_documents)
    store = ChromaVectorStore(path=None, collection_name="togaf_builtin_corpus")
    embedder = cached_embedder(model_name)
    store.replace_index(chunks, embedder)
    file_counts = Counter(str(chunk.metadata["source"]) for chunk in chunks)
    return store, len(source_documents), dict(file_counts)


def render_answer(entry: dict) -> None:
    answer = entry.get("answer", {})
    st.markdown(f"**{answer.get('tema', 'Respuesta')}**")
    st.write(answer.get("respuesta", entry.get("content", "")))
    example = answer.get("ejemplo", "")
    if example:
        st.markdown(f"**Ejemplo:** {example}")
    if answer.get("nivel"):
        st.caption(f"Nivel: {answer['nivel']}")

    sources = entry.get("sources", [])
    if sources:
        unique_documents = len({source.get("source", "Documento") for source in sources})
        with st.expander(
            f"Evidencia: {len(sources)} fragmentos de {unique_documents} documentos"
        ):
            for index, source in enumerate(sources, start=1):
                location = source.get("source", "Documento")
                page = source.get("page")
                if page is not None:
                    location += f" · página {page}"
                chunk_index = source.get("chunk_index")
                if chunk_index is not None:
                    location += f" · fragmento {int(chunk_index) + 1}"
                st.markdown(f"**{index}. {location}**")
                st.caption(f"Similitud: {source.get('similarity', 0):.3f}")
                st.write(source.get("text", ""))


st.title("Tutor TOGAF")
st.caption("Asistente académico con recuperación de fragmentos y fuentes visibles.")

if "messages" not in st.session_state:
    st.session_state.messages = []

embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
try:
    rag_store, source_page_count, indexed_file_counts = load_builtin_index(embedding_model)
except Exception as exc:
    logging.exception("No se pudo preparar el corpus TOGAF incluido")
    st.error(f"No se pudo cargar la base de conocimiento incluida: {exc}")
    st.stop()

with st.sidebar:
    st.header("Base de conocimiento")
    st.success(
        f"Corpus TOGAF listo: {len(indexed_file_counts)} documentos, "
        f"{source_page_count} páginas y {rag_store.count()} fragmentos."
    )
    with st.expander("Ver documentos incluidos"):
        for filename, chunk_count in sorted(indexed_file_counts.items()):
            st.write(f"{filename}: {chunk_count} fragmentos")

    st.divider()
    top_k = st.slider(
        "Fragmentos máximos por pregunta (top_k)",
        min_value=1,
        max_value=8,
        value=5,
    )
    if st.button("Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "El corpus TOGAF autorizado viene incluido con el tutor y se indexa al iniciar. "
        "Solo la pregunta, el historial reciente y los fragmentos recuperados se envían a Groq. "
        "Consulta la [política de datos de Groq](https://console.groq.com/docs/your-data)."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_answer(message)
        else:
            st.write(message["content"])

question = st.chat_input("Pregunta sobre TOGAF…")
if question:
    api_key = groq_api_key()
    if not api_key:
        st.error("Falta configurar GROQ_API_KEY en el entorno o en los secretos de Streamlit.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    embedder = cached_embedder(embedding_model)
    pipeline = RAGPipeline(
        rag_store,
        embedder,
        min_similarity=float(os.getenv("MIN_RETRIEVAL_SIMILARITY", "0.55")),
        model_name=model_name,
    )
    history = st.session_state.messages[:-1]
    client = cached_groq_client(api_key)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Buscando evidencia en los documentos…"):
                answer, retrieved = pipeline.answer(
                    question,
                    history,
                    top_k=top_k,
                    client=client,
                )
            source_entries = [
                {
                    "source": str(item.metadata.get("source", "Documento")),
                    "page": item.metadata.get("page"),
                    "chunk_index": item.metadata.get("chunk_index"),
                    "similarity": item.similarity,
                    "text": item.text,
                }
                for item in retrieved
            ]
            entry = {
                "role": "assistant",
                "content": answer.respuesta,
                "answer": answer.model_dump(),
                "sources": source_entries,
            }
            render_answer(entry)
        st.session_state.messages.append(entry)
    except Exception as exc:
        st.error(f"No se pudo responder esta pregunta: {exc}")
