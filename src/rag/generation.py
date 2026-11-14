from __future__ import annotations

import unicodedata
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from prompts.rag_evaluation_prompt import RAG_EVALUATION_SYSTEM_PROMPT
from src.rag.vector_store import RetrievedChunk

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class TutorAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tema: str = Field(description="Concepto principal de la respuesta")
    respuesta: str = Field(description="Explicación sustentada en el contexto recuperado")
    ejemplo: str = Field(description="Ejemplo breve o una cadena vacía si no aplica")
    nivel: Literal["básico", "intermedio", "avanzado"]


def _asks_for_adm_phase_list(question: str) -> bool:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", question.casefold())
        if not unicodedata.combining(character)
    )
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    has_phase_term = bool(tokens.intersection({"fase", "fases", "phase", "phases"}))
    asks_for_sequence = any(
        marker in normalized
        for marker in (
            "secuencia", "sequence", "cuales son las fases", "what are the phases",
            "enumera las fases", "list the phases", "list of phases",
        )
    )
    asks_about_requirements = any(
        marker in normalized
        for marker in ("gestion de requisitos", "requirements management", "requisitos", "requerimientos")
    )
    return "adm" in tokens and has_phase_term and asks_for_sequence and not asks_about_requirements


def _has_adm_phase_sequence_evidence(chunks: list[RetrievedChunk]) -> bool:
    required_labels = (
        "figure 5-1",
        "preliminary phase",
        "phase a",
        "phase b",
        "phase c",
        "phase d",
        "phase e",
        "phase f",
        "phase g",
        "phase h",
        "requirements management",
    )
    return any(
        all(label in chunk.text.casefold() for label in required_labels)
        for chunk in chunks
    )


def _deterministic_adm_phase_list_answer() -> TutorAnswer:
    return TutorAnswer(
        tema="ADM y secuencia de fases",
        respuesta=(
            "El ADM es el método de TOGAF para desarrollar la arquitectura empresarial. "
            "La Figura 5-1 del documento muestra primero la Fase Preliminar y luego el "
            "ciclo de fases: A (Visión de Arquitectura), B (Arquitectura de Negocio), "
            "C (Arquitecturas de Sistemas de Información), D (Arquitectura Tecnológica), "
            "E (Oportunidades y Soluciones), F (Planificación de la Migración), "
            "G (Gobernanza de la Implementación) y H (Gestión del Cambio de Arquitectura). "
            "La Gestión de Requisitos aparece en el centro conectada con las fases; no es "
            "una Fase I. El corpus muestra la secuencia, pero no describe con detalle todas "
            "las actividades de las fases D a H."
        ),
        ejemplo="",
        nivel="intermedio",
    )


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        source = str(chunk.metadata.get("source", "Documento sin nombre"))
        page = chunk.metadata.get("page")
        location = f"{source}, página {page}" if page is not None else source
        blocks.append(f"[Fragmento {index} | {location}]\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_message(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
) -> str:
    normalized_question = "".join(
        character
        for character in unicodedata.normalize("NFKD", question.casefold())
        if not unicodedata.combining(character)
    )
    intent_guidance: list[str] = []
    if any(term in normalized_question for term in ("gestion de requisitos", "requirements management")):
        intent_guidance.append(
            "Para una pregunta sobre gestión de requisitos y ciclo ADM, explica la relación "
            "que describan los fragmentos y limita cada afirmación a lo que dicen. Si el "
            "contexto presenta el Repositorio de Requisitos como una opción, descríbelo como "
            "opcional. No afirmes que versiona requisitos, alimenta cada fase, asegura "
            "trazabilidad o propaga cambios automáticamente, salvo que el texto lo diga. "
            "No la conviertas en una lista genérica de fases."
        )
    if _asks_for_adm_phase_list(question):
        intent_guidance.append(
            "Si los fragmentos muestran la secuencia de fases del ADM, enumérala en el orden "
            "correcto: fase preliminar primero y después las fases A, B, C, D, E, F, G y H. "
            "La Gestión de Requisitos es transversal y no una fase numerada. Distingue las fases "
            "que solo aparecen nombradas de las que sí tienen actividades u objetivos descritos; "
            "no atribuyas actividades genéricas a todas. Nunca repitas la pregunta como respuesta."
        )
    compared_domains = sum(
        term in normalized_question
        for term in ("negocio", "datos", "aplicaciones", "tecnologic", "business", "data", "application", "technology")
    )
    if compared_domains >= 2:
        intent_guidance.append(
            "Para comparar dominios, responde de forma parcial y separada: explica cada "
            "dominio que los fragmentos describan y nombra cuáles solo aparecen mencionados "
            "sin definición. No rechaces toda la comparación si hay evidencia útil para "
            "algunos dominios, y no infieras funciones a partir de un nombre solamente."
        )
    if any(term in normalized_question for term in ("entregable", "deliverable", "outputs", "productos de trabajo")):
        intent_guidance.append(
            "En preguntas sobre entregables, presenta solo las fases y productos que estén "
            "en los fragmentos y aclara que la lista es parcial si no aparecen todas las "
            "fases del ADM. No atribuyas validación de cada entregable al Architecture "
            "Board a menos que el texto lo diga expresamente."
        )

    history_lines: list[str] = []
    for item in (history or [])[-4:]:
        role = "Estudiante" if item.get("role") == "user" else "Tutor"
        content = str(item.get("content", "")).strip()
        if content:
            history_lines.append(f"{role}: {content[:700]}")
    history_text = "\n".join(history_lines) or "Sin turnos anteriores."
    context_text = format_context(chunks) or "No se recuperó evidencia relevante."
    return (
        "<INSTRUCCIONES>\n"
        "Responde la pregunta actual aplicando el System Prompt. Usa el historial solo "
        "para resolver referencias a turnos anteriores.\n"
        "Si los fragmentos contienen evidencia pertinente, redacta una respuesta directa. "
        "Nunca devuelvas una respuesta vacía ni repitas la pregunta como respuesta. Si no "
        "hay evidencia suficiente, indícalo claramente.\n"
        "Si el contexto respalda parte de la pregunta, responde esa parte y señala con "
        "precisión qué falta; no rechaces toda la pregunta. No completes vacíos con "
        "conocimiento general. Cada dato factual debe estar respaldado por los fragmentos; "
        "no deduzcas contenido a partir del título del archivo ni del nombre de una fase. "
        "El campo ejemplo debe quedar vacío salvo que algún fragmento respalde explícitamente "
        "ese ejemplo.\n"
        + "\n".join(intent_guidance)
        + "\n"
        "</INSTRUCCIONES>\n"
        f"<HISTORIAL>\n{history_text}\n</HISTORIAL>\n"
        f"<CONTEXTO>\n{context_text}\n</CONTEXTO>\n"
        f"<PREGUNTA>\n{question.strip()}\n</PREGUNTA>"
    )


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
    *,
    client=None,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> TutorAnswer:
    """Generate one schema-constrained answer; only retrieved chunks are sent."""
    if _asks_for_adm_phase_list(question) and _has_adm_phase_sequence_evidence(chunks):
        # The ADM sequence is transcribed from a labeled source figure. Return it
        # directly when that complete evidence is retrieved to prevent phase-order
        # hallucinations and avoid spending a Groq call on a fixed list.
        return _deterministic_adm_phase_list_answer()

    if client is None:
        from groq import Groq

        client = Groq()

    request_options = {}
    if model_name in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
        # Keep internal reasoning small so the output budget is available for the
        # required JSON object, especially on multi-phase ADM questions.
        request_options["reasoning_effort"] = "low"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": RAG_EVALUATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_message(question, chunks, history),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tutor_answer",
                "strict": True,
                "schema": TutorAnswer.model_json_schema(),
            },
        },
        temperature=0.2,
        max_completion_tokens=2048,
        **request_options,
    )
    raw_text = response.choices[0].message.content
    if not raw_text:
        raise ValueError("Groq devolvió una respuesta vacía.")
    return TutorAnswer.model_validate_json(raw_text)
