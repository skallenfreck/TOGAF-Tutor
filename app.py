from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import uuid
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag.documents import chunk_documents, extract_documents
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


def fingerprint_uploads(files: list[tuple[str, bytes]]) -> str | None:
    if not files:
        return None
    file_fingerprints = []
    for filename, data in files:
        digest = hashlib.sha256()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        file_fingerprints.append(digest.hexdigest())
    combined = "\n".join(sorted(file_fingerprints)).encode("ascii")
    return hashlib.sha256(combined).hexdigest()


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
if "rag_store" not in st.session_state:
    st.session_state.rag_store = None
if "corpus_fingerprint" not in st.session_state:
    st.session_state.corpus_fingerprint = None
if "indexed_file_counts" not in st.session_state:
    st.session_state.indexed_file_counts = {}

with st.sidebar:
    st.header("Base de conocimiento")
    uploads = st.file_uploader(
        "Carga los documentos que tienes autorización para procesar",
        type=["pdf", "docx", "txt", "html", "htm"],
        accept_multiple_files=True,
        help="Los documentos se procesan en la sesión y no se guardan en el índice local del servidor.",
    )
    uploaded_files = [(upload.name, upload.getvalue()) for upload in (uploads or [])]
    uploaded_fingerprint = fingerprint_uploads(uploaded_files)
    if any(Path(filename).suffix.lower() == ".pdf" for filename, _ in uploaded_files):
        if importlib.util.find_spec("fontTools") is None:
            st.warning(
                "Falta fontTools. Instálalo con «python -m pip install -r requirements.txt» "
                "y vuelve a crear el índice para evitar errores al leer algunos PDF."
            )
    if st.button("Crear índice de esta sesión", type="primary", disabled=not uploads):
        try:
            source_documents = []
            for filename, data in uploaded_files:
                source_documents.extend(extract_documents(data, filename))
            chunks = chunk_documents(source_documents)
            store = ChromaVectorStore(
                path=None,
                collection_name=f"session_{uuid.uuid4().hex[:16]}",
            )
            model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            embedder = cached_embedder(model_name)
            with st.spinner("Extrayendo texto y creando el índice local…"):
                store.replace_index(chunks, embedder)
            st.session_state.rag_store = store
            st.session_state.corpus_fingerprint = uploaded_fingerprint
            st.session_state.indexed_file_counts = dict(
                Counter(str(chunk.metadata["source"]) for chunk in chunks)
            )
            st.session_state.messages = []
            st.success(f"Índice listo: {len(source_documents)} páginas/documentos y {len(chunks)} fragmentos.")
        except Exception as exc:
            logging.exception("No se pudo crear el índice de esta sesión")
            st.error(f"No se pudo crear el índice: {exc}")

    index_is_current = (
        st.session_state.rag_store is not None
        and uploaded_fingerprint == st.session_state.corpus_fingerprint
    )
    if st.session_state.rag_store is not None and index_is_current:
        indexed_files = st.session_state.indexed_file_counts or {
            filename: 0 for filename, _ in uploaded_files
        }
        file_count = len(indexed_files)
        st.success(
            f"Índice de esta sesión: {file_count} archivos, "
            f"{st.session_state.rag_store.count()} fragmentos"
        )
        with st.expander("Ver archivos incluidos en el índice"):
            for filename, chunk_count in sorted(indexed_files.items()):
                if chunk_count:
                    st.write(f"{filename}: {chunk_count} fragmentos")
                else:
                    st.write(filename)
            if not st.session_state.indexed_file_counts:
                st.caption("Pulsa «Crear índice» otra vez para mostrar el total de fragmentos por archivo.")
    elif st.session_state.rag_store is not None:
        st.warning("Cambió la lista de documentos. Pulsa «Crear índice de esta sesión» para incluir los archivos nuevos antes de preguntar.")
    else:
        st.info("Carga tus documentos de TOGAF y crea el índice para empezar.")

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
        "La aplicación recibe los archivos cargados para procesarlos en la sesión. "
        "Solo los fragmentos recuperados, la pregunta y el historial reciente se envían a Groq. "
        "Consulta la [política de datos de Groq](https://console.groq.com/docs/your-data) "
        "antes de cargar documentos privados."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_answer(message)
        else:
            st.write(message["content"])

question = st.chat_input("Pregunta sobre los documentos TOGAF cargados…")
if question:
    if not index_is_current:
        st.warning("Primero carga los PDF y crea el índice de esta sesión.")
        st.stop()

    api_key = groq_api_key()
    if not api_key:
        st.error("Falta configurar GROQ_API_KEY en el entorno o en los secretos de Streamlit.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    embedder = cached_embedder(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    pipeline = RAGPipeline(
        st.session_state.rag_store,
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
