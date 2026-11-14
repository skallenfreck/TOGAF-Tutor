from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from prompts.system_prompt import SYSTEM_PROMPT
from src.agent.tools import BuscarEnBaseTool, BuscarEnWebTool, GenerarReferenciaApaTool
from src.rag.generation import DEFAULT_GROQ_MODEL
from src.rag.pipeline import ABSTENTION_TEXT, RAGPipeline

MAX_TOOL_ITERATIONS = 4
MAX_TOOL_CALLS_PER_ITERATION = 3
MAX_HISTORY_MESSAGES = 8
DEFAULT_TOOL_LOG = Path("logs/tool_usage.jsonl")
_LOG_LOCK = threading.Lock()


class AgentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tema: str
    respuesta: str
    ejemplo: str
    nivel: Literal["básico", "intermedio", "avanzado"]
    fuentes: list[str]


@dataclass
class AgentRun:
    answer: AgentAnswer
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    llm_call_count: int = 0
    iterations: int = 0


def _casual_message(question: str) -> bool:
    normalized = re.sub(r"[^a-záéíóúñü0-9 ]", "", question.casefold()).strip()
    normalized = "".join(
        char for char in unicodedata.normalize("NFD", normalized)
        if unicodedata.category(char) != "Mn"
    )
    casual_phrases = {
        "hola", "hola tutor", "hola como estas", "buenos dias", "buenas tardes", "buenas noches",
        "gracias", "muchas gracias", "chao", "adios", "como estas", "que tal",
    }
    return normalized in casual_phrases


def _tool_schemas(*, include_web: bool = False) -> list[dict[str, Any]]:
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "buscar_en_base",
                "description": BuscarEnBaseTool.description,
                "parameters": {
                    "type": "object",
                    "properties": {"consulta": {"type": "string", "description": "Pregunta o subpregunta para buscar en el corpus TOGAF."}},
                    "required": ["consulta"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generar_referencia_apa",
                "description": GenerarReferenciaApaTool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "documento": {"type": "string", "description": "Nombre exacto o reconocible del PDF del corpus."},
                        "pagina": {"type": ["integer", "null"], "description": "Página a citar, si se conoce."},
                    },
                    "required": ["documento"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    if include_web:
        schemas.append({
            "type": "function",
            "function": {
                "name": "buscar_en_web",
                "description": BuscarEnWebTool.description,
                "parameters": {
                    "type": "object",
                    "properties": {"consulta_publica": {"type": "string", "description": "Consulta pública sin documentos ni fragmentos del corpus."}},
                    "required": ["consulta_publica"],
                    "additionalProperties": False,
                },
            },
        })
    return schemas


def _append_tool_log(
    path: Path,
    *,
    question: str,
    trace: list[dict[str, Any]],
    llm_call_count: int,
    iterations: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interaction_id": uuid.uuid4().hex,
        "question": question,
        "tools": [{"name": item["name"], "arguments": item["arguments"]} for item in trace],
        "tool_call_count": len(trace),
        "llm_call_count": llm_call_count,
        "tool_iterations": iterations,
    }
    with _LOG_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _collect_sources(tool_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    if tool_name == "buscar_en_base":
        return [
            {
                "source": item["documento"],
                "origin": "base",
                "page": item.get("pagina"),
                "section": item.get("seccion"),
                "similarity": item.get("similitud", 0.0),
                "text": item["fragmento"],
            }
            for item in result.get("resultados", [])
        ]
    if tool_name == "generar_referencia_apa":
        return [{"source": result["documento"], "origin": "base", "page": result.get("pagina"), "citation": result["referencia_apa"]}]
    if tool_name == "buscar_en_web":
        return [{"source": item.get("titulo", "Fuente web"), "origin": item.get("origen", "web"), "url": item.get("url", ""), "citation": item.get("url", ""), "text": item.get("extracto", "")}
                for item in result.get("resultados", []) if item.get("url")]
    return []


def _dedupe_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for item in items:
        key = (str(item.get("source", "")), str(item.get("page", "")), str(item.get("citation", "")))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def run_agent(
    question: str,
    history: list[dict[str, str]] | None,
    pipeline: RAGPipeline,
    apa_tool: GenerarReferenciaApaTool,
    *,
    client: Any,
    model_name: str = DEFAULT_GROQ_MODEL,
    tool_log_path: str | Path = DEFAULT_TOOL_LOG,
) -> AgentRun:
    """Run a bounded tool-calling loop and log tool arguments and model calls."""
    search_tool = BuscarEnBaseTool(pipeline)
    web_tool = BuscarEnWebTool()
    message_history = [
        {"role": item.get("role", "user"), "content": str(item.get("content", ""))[:1500]}
        for item in (history or [])[-MAX_HISTORY_MESSAGES:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *message_history,
        {"role": "user", "content": question.strip()},
    ]
    trace: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    llm_calls = 0
    iterations = 0
    last_content = ""
    casual = _casual_message(question)
    if casual:
        answer = AgentAnswer(
            tema="Saludo",
            respuesta="¡Hola! Estoy aquí para ayudarte con TOGAF y arquitectura empresarial. ¿Qué te gustaría consultar?",
            ejemplo="",
            nivel="básico",
            fuentes=[],
        )
        _append_tool_log(Path(tool_log_path), question=question, trace=trace, llm_call_count=0, iterations=0)
        return AgentRun(answer=answer, sources=[], tool_trace=[], llm_call_count=0, iterations=0)
    web_enabled = False
    apa_tool_error: str | None = None
    request_options = {"reasoning_effort": "low"} if model_name.startswith("openai/gpt-oss-") else {}

    for _ in range(MAX_TOOL_ITERATIONS):
        tool_options = (
            {} if casual else {"tools": _tool_schemas(include_web=web_enabled), "tool_choice": "auto"}
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **tool_options,
            temperature=0.2,
            max_completion_tokens=2048,
            **request_options,
        )
        llm_calls += 1
        message = response.choices[0].message
        last_content = message.content or ""
        tool_calls = message.tool_calls or []
        if not tool_calls:
            break

        iterations += 1
        serialized_calls = [call.model_dump(exclude_none=True) for call in tool_calls]
        messages.append({"role": "assistant", "content": message.content, "tool_calls": serialized_calls})
        for index, call in enumerate(tool_calls):
            if index >= MAX_TOOL_CALLS_PER_ITERATION or len(trace) >= MAX_TOOL_ITERATIONS:
                tool_result: dict[str, Any] = {"error": "Se alcanzó el límite de llamadas a herramientas para esta pregunta."}
                name = str(call.function.name)
                arguments: dict[str, Any] = {}
            else:
                name = str(call.function.name)
                arguments: dict[str, Any] = {}
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Los argumentos deben ser un objeto JSON.")
                    if name == search_tool.name:
                        tool_result = search_tool.invoke_with_history(arguments, message_history)
                        sources.extend(_collect_sources(name, tool_result))
                        # «Primero la base»: after retrieving evidence, let the model
                        # judge whether it actually answers the question. Web is offered
                        # only on the following turn, never before the corpus search.
                        web_enabled = True
                    elif name == apa_tool.name:
                        tool_result = apa_tool.invoke(arguments)
                        sources.extend(_collect_sources(name, tool_result))
                    elif name == web_tool.name:
                        if not web_enabled:
                            raise ValueError("La búsqueda web solo se habilita si la base no devuelve resultados.")
                        # Only the student's public question reaches the web tool;
                        # model-supplied arguments and retrieved corpus text are not forwarded.
                        arguments = {"consulta_publica": question.strip()[:300]}
                        tool_result = web_tool.invoke(arguments)
                        sources.extend(_collect_sources(name, tool_result))
                    else:
                        tool_result = {"error": f"Herramienta no permitida: {name}"}
                except Exception as exc:
                    logging.exception("Falló la herramienta %s", name)
                    tool_result = {"error": str(exc)}
                    arguments = arguments if isinstance(arguments, dict) else {}
                    if name == apa_tool.name:
                        apa_tool_error = str(exc)
            trace.append({"name": name, "arguments": arguments})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })
        if apa_tool_error:
            break

    sources = _dedupe_sources(sources)
    has_task_result = any(item["name"] == "generar_referencia_apa" for item in trace)
    has_search_result = any(item["name"] == "buscar_en_base" for item in trace)
    has_web_result = any(item["name"] == "buscar_en_web" for item in trace)
    has_base_evidence = any(item.get("origin") == "base" for item in sources)
    has_web_evidence = any(item.get("origin") == "web" for item in sources)

    # A factual answer cannot bypass tools and fall back to model memory. Greetings
    # and completed deterministic tasks are the only no-search exceptions.
    if apa_tool_error:
        if "no pertenece al corpus" in apa_tool_error.casefold():
            response = "Ese documento no pertenece al corpus TOGAF incluido, así que no puedo generar una referencia APA verificable."
        else:
            response = "No puedo generar una referencia APA verificable con los datos indicados."
        answer = AgentAnswer(tema="Referencia APA no disponible", respuesta=response, ejemplo="", nivel="básico", fuentes=[])
    elif not has_search_result and not has_task_result and not has_web_result and not casual:
        answer = AgentAnswer(tema="Información no encontrada en el corpus", respuesta=ABSTENTION_TEXT, ejemplo="", nivel="básico", fuentes=[])
    elif ((has_search_result and not has_base_evidence and not has_task_result and not has_web_evidence)
          or (has_web_result and not has_web_evidence and not has_task_result and not has_base_evidence)):
        answer = AgentAnswer(tema="Información no encontrada en el corpus", respuesta=ABSTENTION_TEXT, ejemplo="", nivel="básico", fuentes=[])
    else:
        # If the model produced a final answer without tools and followed JSON, reuse
        # it; after tool use, synthesize once with tools disabled and the tool outputs
        # in the conversation so every factual claim is grounded in returned data.
        parsed = None
        if not trace and last_content:
            try:
                parsed = AgentAnswer.model_validate_json(last_content)
            except Exception:
                parsed = None
        if parsed is None:
            final_response = client.chat.completions.create(
                model=model_name,
                messages=messages + [{"role": "system", "content": "Redacta ahora la respuesta final; no llames herramientas. Usa solo resultados de herramientas ya recibidos."}],
                response_format={"type": "json_schema", "json_schema": {"name": "tutor_agent_answer", "strict": True, "schema": AgentAnswer.model_json_schema()}},
                temperature=0.2,
                max_completion_tokens=2048,
                **request_options,
            )
            llm_calls += 1
            final_text = final_response.choices[0].message.content or ""
            parsed = AgentAnswer.model_validate_json(final_text)
        parsed.fuentes = [
            (f"[web: {item['url']}]" if item.get("origin") == "web" and item.get("url") else
             f"[base: {item['source']}, página {item['page']}]" if item.get("page") is not None else
             f"[base: {item.get('citation', item['source'])}]")
            for item in sources
        ]
        answer = parsed

    _append_tool_log(Path(tool_log_path), question=question, trace=trace, llm_call_count=llm_calls, iterations=iterations)
    return AgentRun(answer=answer, sources=sources, tool_trace=trace, llm_call_count=llm_calls, iterations=iterations)
