from __future__ import annotations

import logging
import json
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
from src.agent.runner import run_agent
from src.agent.tools import GenerarReferenciaApaTool

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
    render_tool_usage(entry)
    render_sources(entry)


def render_tool_usage(entry: dict) -> None:
    trace = entry.get("tool_trace", [])
    call_count = int(entry.get("llm_call_count", 0) or 0)
    with st.expander(f"Herramientas usadas ({len(trace)}) · llamadas al LLM ({call_count})"):
        if not trace:
            st.caption("El agente respondió sin llamar herramientas.")
        else:
            tool_labels = {
                "buscar_en_base": "Búsqueda en la base de conocimiento",
                "buscar_en_web": "Búsqueda web controlada",
                "generar_referencia_apa": "Generación de referencia APA",
            }
            for index, call in enumerate(trace, start=1):
                name = call.get("name", "herramienta")
                st.markdown(f"**{index}. {tool_labels.get(name, name)}**")
                arguments = call.get("arguments", {})
                st.code(json.dumps(arguments, ensure_ascii=False, indent=2), language="json")


def render_sources(entry: dict) -> None:
    sources = entry.get("sources", [])
    base_sources = [source for source in sources if source.get("origin", "base") == "base"]
    web_sources = [source for source in sources if source.get("origin") == "web"]
    if not base_sources and not web_sources:
        st.caption("No se encontraron fuentes suficientes en la base ni en los sitios web permitidos.")
        return

    if base_sources:
        with st.expander(f"Fuentes de la base de conocimiento ({len(base_sources)})"):
            for index, source in enumerate(base_sources, start=1):
                document = source.get("source", "Documento")
                page = source.get("page")
                section = source.get("section")
                location = f"página {page}" if page is not None else str(section or "sección no especificada")
                st.markdown(f"**{index}. [base: {document}, {location}]**")
                if source.get("citation"):
                    st.write(source["citation"])
                if source.get("similarity") is not None:
                    st.caption(f"Similitud: {float(source['similarity']):.3f}")
                if source.get("text"):
                    st.write(source["text"])

    if web_sources:
        with st.expander(f"Fuentes web ({len(web_sources)})"):
            for index, source in enumerate(web_sources, start=1):
                title = source.get("source", "Resultado web")
                url = source.get("url", "")
                st.markdown(f"**{index}. {title}**")
                if url:
                    st.markdown(f"[web: {url}]({url})")
                if source.get("text"):
                    st.write(source["text"])


st.title("Tutor TOGAF")
st.caption("Agente académico TOGAF con herramientas, historial conversacional y fuentes visibles.")

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
    if st.button("Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "El corpus TOGAF autorizado viene incluido con el tutor y se indexa al iniciar. "
        "Solo la pregunta, el historial reciente y los fragmentos recuperados se envían a Groq. "
        "Para consultas públicas actuales, la búsqueda web recibe solo la pregunta, nunca los fragmentos. "
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
    apa_tool = GenerarReferenciaApaTool(Path(__file__).resolve().parent / "data" / "source_documents")
    try:
        with st.chat_message("assistant"):
            with st.spinner("Decidiendo qué herramienta usar…"):
                agent_run = run_agent(
                    question,
                    history,
                    pipeline,
                    apa_tool,
                    client=client,
                    model_name=model_name,
                    tool_log_path=os.getenv("TOOL_USAGE_LOG", "logs/tool_usage.jsonl"),
                )
            answer = agent_run.answer
            entry = {
                "role": "assistant",
                "content": answer.respuesta,
                "answer": answer.model_dump(),
                "sources": agent_run.sources,
                "tool_trace": agent_run.tool_trace,
                "llm_call_count": agent_run.llm_call_count,
            }
            render_answer(entry)
        st.session_state.messages.append(entry)
    except Exception as exc:
        st.error(f"No se pudo responder esta pregunta: {exc}")
