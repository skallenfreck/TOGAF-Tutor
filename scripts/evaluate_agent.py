from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

from scripts.evaluate_ragas import (
    METRIC_COLUMNS,
    FastEmbedRagasEmbedding,
    _append_checkpoint,
    _load_checkpoint,
    _load_dataset,
    _looks_like_abstention,
    _patch_optional_vertexai_import,
)
from src.agent.runner import run_agent
from src.agent.tools import ALLOWED_WEB_DOMAINS, GenerarReferenciaApaTool
from src.rag.documents import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SUPPORTED_SUFFIXES,
    chunk_documents,
    load_directory,
)
from src.rag.embeddings import DEFAULT_EMBEDDING_MODEL, get_local_embedder
from src.rag.generation import DEFAULT_GROQ_MODEL
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import ChromaVectorStore

_patch_optional_vertexai_import()
from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "avance3_agent"
DEFAULT_BASELINE_METRICS = ROOT / "evaluation" / "avance2_baseline_metrics.csv"


class _TrackedCompletions:
    def __init__(self, owner: "TrackedGroqClient") -> None:
        self.owner = owner

    def create(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            response = self.owner._client.chat.completions.create(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt = getattr(usage, "prompt_tokens", None)
                completion = getattr(usage, "completion_tokens", None)
                total = getattr(usage, "total_tokens", None)
                self.owner.prompt_tokens += int(prompt or 0)
                self.owner.completion_tokens += int(completion or 0)
                self.owner.total_tokens += int(total if total is not None else (prompt or 0) + (completion or 0))
                self.owner.usage_records += 1
            return response
        finally:
            self.owner.call_count += 1
            self.owner.elapsed_seconds += time.perf_counter() - started


class _TrackedChat:
    def __init__(self, owner: "TrackedGroqClient") -> None:
        self.completions = _TrackedCompletions(owner)


class TrackedGroqClient:
    """Transparent Groq client proxy that counts generation tokens and latency."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = _TrackedChat(self)
        self.reset()

    def reset(self) -> None:
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.usage_records = 0
        self.elapsed_seconds = 0.0

    def snapshot(self, answer_elapsed: float) -> dict[str, Any]:
        return {
            "llm_call_count": self.call_count,
            "prompt_tokens": self.prompt_tokens if self.usage_records else None,
            "completion_tokens": self.completion_tokens if self.usage_records else None,
            "total_tokens": self.total_tokens if self.usage_records else None,
            "usage_records": self.usage_records,
            "llm_elapsed_seconds": round(self.elapsed_seconds, 4),
            "response_time_seconds": round(answer_elapsed, 4),
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"No se encontró el archivo requerido: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _corpus_hash(documents_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item for item in documents_dir.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    ):
        digest.update(path.relative_to(documents_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evaluation_signature(questions: list[dict], model: str, corpus_hash: str, embedding: str) -> str:
    try:
        ragas_version = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError:
        ragas_version = "unknown"
    payload = {
        "questions": questions,
        "model": model,
        "corpus_hash": corpus_hash,
        "embedding_model": embedding,
        "ragas_version": ragas_version,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_CHUNK_OVERLAP,
        "baseline_top_k": 3,
        "agent_tool_limits": {"max_fragments": 5, "max_fragment_chars": 900},
        "metrics": METRIC_COLUMNS,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _prepare_output(output_dir: Path, signature: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "checkpoint_metadata.json"
    answers_path = output_dir / "answers_checkpoint.jsonl"
    metrics_path = output_dir / "metrics_checkpoint.jsonl"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            raise SystemExit("Los checkpoints pertenecen a otra configuración. Usa otra carpeta --output.")
    elif answers_path.exists() or metrics_path.exists():
        raise SystemExit("Hay checkpoints sin metadatos; usa otra carpeta --output para evitar mezclar datos.")
    else:
        metadata_path.write_text(json.dumps({"signature": signature}, indent=2), encoding="utf-8")
    return answers_path, metrics_path


def _answer_record(question: dict, variant: str, answer: str, contexts: list[str], details: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "variant": variant,
        "question_id": str(question["id"]),
        "user_input": question["question"],
        "response": answer,
        "reference": question["reference"],
        "retrieved_contexts": contexts,
        "expects_abstention": bool(question.get("expects_abstention", False)),
        "did_abstain": _looks_like_abstention(answer),
        **details,
        **(extra or {}),
    }


def _mean(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if value not in (None, "")]
    return sum(nums) / len(nums) if nums else None


def _write_analysis(path: Path, metrics: list[dict[str, Any]], costs: list[dict[str, Any]], behaviors: list[dict[str, Any]]) -> None:
    baseline = next(row for row in metrics if row["variant"].startswith("pipeline_"))
    agent = next(row for row in metrics if row["variant"] == "agent")
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer relevancy",
        "context_precision": "Context precision",
        "context_recall": "Context recall",
    }
    lines = [
        "# Análisis: RAG fijo frente a agente",
        "",
        "Las métricas Ragas se calcularon sobre las mismas preguntas de evaluación. La configuración fija corresponde a top_k=3; el agente usa los fragmentos que devolvieron sus tools. Los cambios describen esta ejecución y no prueban causalidad por sí solos.",
        "",
        "| Métrica | RAG fijo | Agente | Diferencia |",
        "|---|---:|---:|---:|",
    ]
    deltas = {}
    for key, label in labels.items():
        left = baseline.get(key)
        right = agent.get(key)
        delta = (float(right) - float(left)) if left is not None and right is not None else None
        deltas[key] = delta
        lines.append(f"| {label} | {float(left):.3f} | {float(right):.3f} | {delta:+.3f} |" if delta is not None else f"| {label} | — | — | — |")
    changed = [(key, value) for key, value in deltas.items() if value is not None and abs(value) >= 0.001]
    improved = [labels[key] for key, value in changed if value > 0]
    declined = [labels[key] for key, value in changed if value < 0]
    lines += ["", "## Interpretación", ""]
    lines.append("Mejoraron: " + (", ".join(improved) if improved else "ninguna métrica tuvo una mejora de al menos 0.001") + ".")
    lines.append("Empeoraron: " + (", ".join(declined) if declined else "ninguna métrica tuvo una disminución de al menos 0.001") + ".")
    lowest = min((key for key in labels if agent.get(key) is not None), key=lambda key: float(agent[key]))
    attribution = {
        "faithfulness": "apunta principalmente a la generación y a cuánto respeta la respuesta la evidencia recuperada",
        "answer_relevancy": "apunta a la selección de herramientas, la pertinencia de los fragmentos y la síntesis final",
        "context_precision": "apunta al ranking y filtrado de la recuperación: entró evidencia menos pertinente de la necesaria",
        "context_recall": "apunta a cobertura de la recuperación: faltó traer evidencia necesaria para la respuesta de referencia",
    }[lowest]
    lines.append(f"La métrica más baja del agente fue **{labels[lowest]}** ({float(agent[lowest]):.3f}); su componente candidato de mejora es {attribution}. Esta atribución es una hipótesis diagnóstica y conviene contrastarla con los casos por pregunta.")
    lines.append("Si subió context_recall y bajó context_precision, el agente cubrió más evidencia a costa de incorporar fragmentos menos selectivos. Si bajó faithfulness, revisar el prompt y la síntesis después de varias llamadas a tools; si bajó answer_relevancy, revisar las consultas descompuestas y las abstenciones.")
    lines += ["", "## Costo y comportamiento", "", "| Variante | Llamadas LLM/pregunta | Tokens totales/pregunta | Tiempo de respuesta/pregunta (s) |", "|---|---:|---:|---:|"]
    for row in costs:
        calls = row.get("avg_llm_calls_per_question")
        tokens = row.get("avg_total_tokens_per_question")
        elapsed = row.get("avg_response_time_seconds")
        lines.append(f"| {row['variant']} | {float(calls):.2f} | {float(tokens):.1f} | {float(elapsed):.2f} |" if calls is not None and tokens is not None and elapsed is not None else f"| {row['variant']} | — | — | — |")
    if behaviors:
        passed = sum(bool(row["passed"]) for row in behaviors)
        lines.append(f"\nLa suite de comportamiento cumplió **{passed} de {len(behaviors)} casos**. Los casos fallidos aparecen con detalle en `behavior_results.csv`.")
    lines.append("\nLos tokens y tiempos de esta tabla corresponden a generación de respuestas con Groq; excluyen las llamadas de Ragas usadas como juez.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_behavior(case: dict[str, Any], result: Any) -> tuple[bool, str]:
    names = [item["name"] for item in result.tool_trace]
    answer = result.answer.respuesta.strip()
    lowered = answer.casefold()
    checks = case.get("checks", {})
    problems = []
    for name in case.get("required_tools", []):
        if name not in names:
            problems.append(f"no llamó {name}")
    for name in case.get("forbidden_tools", []):
        if name in names:
            problems.append(f"llamó herramienta prohibida {name}")
    ordered = case.get("ordered_tools", [])
    if ordered:
        positions = [names.index(name) for name in ordered if name in names]
        if len(positions) != len(ordered) or positions != sorted(positions):
            problems.append("no respetó el orden esperado de herramientas")
    if checks.get("must_not_be_abstention") and _looks_like_abstention(answer):
        problems.append("se abstuvo aunque se esperaba una respuesta")
    required_terms = checks.get("answer_contains_all", [])
    if any(term.casefold() not in lowered for term in required_terms):
        problems.append("la respuesta no cubrió todos los temas indicados")
    for tool_name, expectations in checks.get("expected_tool_arguments", {}).items():
        matching = [item.get("arguments", {}) for item in result.tool_trace if item.get("name") == tool_name]
        if not matching:
            problems.append(f"no registró argumentos para {tool_name}")
            continue
        arguments = matching[0]
        for key, expected in expectations.items():
            if key.endswith("_contains"):
                argument_name = key[:-9]
                if str(expected).casefold() not in str(arguments.get(argument_name, "")).casefold():
                    problems.append(f"el argumento {argument_name} no coincide con lo solicitado")
            elif arguments.get(key) != expected:
                problems.append(f"el argumento {key} no coincide con lo solicitado")
    if checks.get("requires_base_citation") and not any(source.get("origin") == "base" for source in result.sources):
        problems.append("faltó evidencia citada de la base")
    if checks.get("requires_web_citation"):
        web_sources = [source for source in result.sources if source.get("origin") == "web"]
        if not web_sources or not any(f"[web: {source.get('url', '')}]" in result.answer.fuentes for source in web_sources):
            problems.append("faltó una cita web")
        allowed = tuple(checks.get("web_domains", ALLOWED_WEB_DOMAINS))
        for source in web_sources:
            host = (urlparse(source.get("url", "")).hostname or "").casefold().removeprefix("www.")
            if not any(host == domain or host.endswith("." + domain) for domain in allowed):
                problems.append("la fuente web no pertenece a los dominios permitidos")
    if checks.get("must_contain_any") and not any(term.casefold() in lowered for term in checks["must_contain_any"]):
        problems.append("no explicó el rechazo del documento fuera del corpus")
    if any(term.casefold() in lowered for term in checks.get("must_not_contain", [])):
        problems.append("inventó metadatos para un documento desconocido")
    if len(answer) > checks.get("max_answer_chars", float("inf")):
        problems.append("la respuesta excedió el máximo de longitud establecido")
    if checks.get("requires_base_citation_or_abstention") and not _looks_like_abstention(answer):
        if not any(source.get("origin") == "base" for source in result.sources):
            problems.append("no incluyó una cita de base ni se abstuvo")
    max_contexts = checks.get("max_contexts")
    if max_contexts is not None and len(result.sources) > max_contexts:
        problems.append("devolvió más contextos que el límite establecido")
    max_chars = checks.get("max_context_chars")
    if max_chars is not None and any(len(source.get("text", "")) > max_chars for source in result.sources):
        problems.append("un contexto excedió el límite de caracteres")
    return not problems, "; ".join(problems) if problems else "Cumplió el comportamiento esperado."


def _run_behavior_cases(cases_path: Path, output_dir: Path, pipeline: RAGPipeline, apa_tool: GenerarReferenciaApaTool, client: TrackedGroqClient, model: str, only_case: str | None = None) -> list[dict[str, Any]]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if len(cases) < 7:
        raise ValueError("La suite de comportamiento debe tener al menos siete casos.")
    selected_cases = [case for case in cases if only_case is None or case["id"] == only_case]
    if only_case is not None and not selected_cases:
        raise SystemExit(f"No existe el caso de comportamiento {only_case}.")
    rows = []
    log_path = output_dir / "behavior_tool_calls.jsonl"
    for index, case in enumerate(selected_cases, 1):
        client.reset()
        started = time.perf_counter()
        try:
            result = run_agent(
                case["question"], [], pipeline, apa_tool, client=client,
                model_name=model, tool_log_path=log_path,
            )
            elapsed = time.perf_counter() - started
            passed, details = _check_behavior(case, result)
            rows.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_behavior": case["expected_behavior"],
                "tools_called": json.dumps([item["name"] for item in result.tool_trace], ensure_ascii=False),
                "tool_arguments": json.dumps([item["arguments"] for item in result.tool_trace], ensure_ascii=False),
                "tool_call_count": len(result.tool_trace),
                "llm_call_count": client.call_count,
                "prompt_tokens": client.prompt_tokens if client.usage_records else None,
                "completion_tokens": client.completion_tokens if client.usage_records else None,
                "total_tokens": client.total_tokens if client.usage_records else None,
                "response_time_seconds": round(elapsed, 4),
                "passed": passed,
                "details": details,
                "answer": result.answer.respuesta,
                "sources": json.dumps(result.answer.fuentes, ensure_ascii=False),
            })
        except Exception as exc:
            rows.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_behavior": case["expected_behavior"],
                "tools_called": "[]",
                "tool_arguments": "[]",
                "tool_call_count": 0,
                "llm_call_count": client.call_count,
                "prompt_tokens": client.prompt_tokens if client.usage_records else None,
                "completion_tokens": client.completion_tokens if client.usage_records else None,
                "total_tokens": client.total_tokens if client.usage_records else None,
                "response_time_seconds": round(time.perf_counter() - started, 4),
                "passed": False,
                "details": f"Ejecución interrumpida: {type(exc).__name__}: {exc}",
                "answer": "",
                "sources": "[]",
            })
        print(f"Caso de comportamiento {case['id']}: {index}/{len(selected_cases)} — {'OK' if rows[-1]['passed'] else 'REVISAR'}", flush=True)
    results_path = output_dir / "behavior_results.csv"
    if only_case is not None and results_path.is_file():
        previous = _read_csv(results_path)
        updated = {row["id"]: row for row in previous}
        updated.update({row["id"]: row for row in rows})
        rows = [updated[case["id"]] for case in cases if case["id"] in updated]
    _write_csv(results_path, rows)
    (output_dir / "behavior_summary.json").write_text(json.dumps({
        "case_count": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
        "expectations_file": str(cases_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el agente frente al RAG fijo con Ragas, casos de comportamiento y telemetría de costo.")
    parser.add_argument("--documents", default=str(ROOT / "data" / "source_documents"))
    parser.add_argument("--dataset", default=str(ROOT / "evaluation" / "questions.json"))
    parser.add_argument("--behavior-cases", default=str(ROOT / "evaluation" / "agent_behavior_cases.json"))
    parser.add_argument("--baseline-metrics", default=str(DEFAULT_BASELINE_METRICS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, help="Piloto opcional; la ejecución formal usa las 18 preguntas.")
    parser.add_argument("--baseline-top-k", type=int, default=3)
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
    parser.add_argument("--skip-behavior", action="store_true", help="Omite los siete casos independientes de comportamiento.")
    parser.add_argument("--only-behavior-case", help="Repite un solo caso y conserva los resultados de los demás.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Falta GROQ_API_KEY en el entorno o en .env.")
    questions = _load_dataset(Path(args.dataset), args.limit)
    documents_dir = Path(args.documents).resolve()
    corpus_hash = _corpus_hash(documents_dir)
    embedding_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    default_embedding_cache = Path(tempfile.gettempdir()) / "fastembed_cache"
    persistent_embedding_cache = ROOT / "data" / "fastembed_cache"
    if default_embedding_cache.is_dir() and any(default_embedding_cache.iterdir()):
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(default_embedding_cache))
    else:
        persistent_embedding_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(persistent_embedding_cache))
    threshold = float(os.getenv("MIN_RETRIEVAL_SIMILARITY", "0.55"))
    output_dir = Path(args.output).resolve()
    signature = _evaluation_signature(questions, args.model, corpus_hash, embedding_name)
    answers_path, metrics_path = _prepare_output(output_dir, signature)

    baseline_rows = _read_csv(Path(args.baseline_metrics))
    baseline_by_id = {
        str(row.get("question_id", row.get("id", ""))): row
        for row in baseline_rows
        if row.get("variant", "baseline_top_k_3") in {"baseline_top_k_3", "pipeline_top_k_3", ""}
    }
    missing = [str(item["id"]) for item in questions if str(item["id"]) not in baseline_by_id]
    if missing:
        raise SystemExit(f"Faltan métricas Ragas del RAG fijo para: {', '.join(missing)}. Revisa --baseline-metrics.")

    from groq import Groq
    from openai import OpenAI

    base_client = Groq(api_key=api_key, timeout=90.0, max_retries=3)
    client = TrackedGroqClient(base_client)
    evaluator_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=90.0, max_retries=3)
    evaluator_llm = llm_factory(args.model, provider="openai", client=evaluator_client, max_tokens=4096, reasoning_effort="low")
    embedder = get_local_embedder(embedding_name)
    evaluator_embeddings = FastEmbedRagasEmbedding(embedder)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(name="context_precision", llm=evaluator_llm),
        LLMContextRecall(name="context_recall", llm=evaluator_llm),
    ]
    chunks = chunk_documents(load_directory(documents_dir), chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
    store = ChromaVectorStore(path=ROOT / "data" / "chroma" / "evaluation", collection_name="togaf_evaluation")
    store.replace_index(chunks, embedder, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
    pipeline = RAGPipeline(store, embedder, min_similarity=threshold, model_name=args.model)
    apa_tool = GenerarReferenciaApaTool(documents_dir)

    answer_cache = _load_checkpoint(answers_path)
    variants = [(f"pipeline_top_k_{args.baseline_top_k}", "pipeline"), ("agent", "agent")]
    answer_rows: list[dict[str, Any]] = []
    expected_total = len(questions) * len(variants)
    completed = 0
    for variant, kind in variants:
        for question in questions:
            key = (variant, str(question["id"]))
            record = answer_cache.get(key)
            if record is None:
                client.reset()
                started = time.perf_counter()
                try:
                    if kind == "pipeline":
                        answer, retrieved = pipeline.answer(question["question"], top_k=args.baseline_top_k, client=client)
                        response = "\n".join(part for part in [answer.respuesta.strip(), f"Ejemplo: {answer.ejemplo.strip()}" if answer.ejemplo.strip() else ""] if part)
                        contexts = [chunk.text for chunk in retrieved]
                        details = client.snapshot(time.perf_counter() - started)
                        record = _answer_record(question, variant, response, contexts, details, {
                            "tool_call_count": 0,
                            "tools_called": [],
                            "retrieved_count": len(retrieved),
                        })
                    else:
                        agent_result = run_agent(
                            question["question"], [], pipeline, apa_tool,
                            client=client, model_name=args.model,
                            tool_log_path=output_dir / "agent_tool_calls.jsonl",
                        )
                        details = client.snapshot(time.perf_counter() - started)
                        contexts = [source.get("text", "") for source in agent_result.sources if source.get("text")]
                        record = _answer_record(question, variant, agent_result.answer.respuesta, contexts, details, {
                            "tool_call_count": len(agent_result.tool_trace),
                            "tools_called": [item["name"] for item in agent_result.tool_trace],
                            "retrieved_count": len(contexts),
                            "sources": agent_result.answer.fuentes,
                        })
                except Exception:
                    print(f"La generación se interrumpió en {variant}/{question['id']}; vuelve a ejecutar el comando para reanudar.", flush=True)
                    raise
                _append_checkpoint(answers_path, record)
                answer_cache[key] = record
            answer_rows.append(record)
            completed += 1
            if completed % max(1, len(questions)) == 0 or completed == expected_total:
                print(f"Respuestas preparadas: {variant} ({completed}/{expected_total})", flush=True)

    cost_rows = []
    for variant, label in variants:
        frame = [row for row in answer_rows if row["variant"] == variant]
        cost_rows.append({
            "variant": variant,
            "question_count": len(frame),
            "avg_llm_calls_per_question": _mean([row.get("llm_call_count") for row in frame]),
            "avg_prompt_tokens_per_question": _mean([row.get("prompt_tokens") for row in frame]),
            "avg_completion_tokens_per_question": _mean([row.get("completion_tokens") for row in frame]),
            "avg_total_tokens_per_question": _mean([row.get("total_tokens") for row in frame]),
            "avg_response_time_seconds": _mean([row.get("response_time_seconds") for row in frame]),
            "avg_llm_elapsed_seconds": _mean([row.get("llm_elapsed_seconds") for row in frame]),
            "questions_with_usage": sum(bool(row.get("usage_records", 0)) for row in frame),
            "tracked_llm_calls_with_usage": sum(int(row.get("usage_records", 0) or 0) for row in frame),
        })
    _write_csv(output_dir / "cost_comparison.csv", cost_rows)

    metric_cache = _load_checkpoint(metrics_path)
    agent_records = [record for record in answer_rows if record["variant"] == "agent"]
    agent_scores = []
    for index, row in enumerate(agent_records, 1):
        key = ("agent", str(row["question_id"]))
        metric_record = metric_cache.get(key)
        if metric_record is None:
            dataset = EvaluationDataset.from_list([{column: row[column] for column in ("user_input", "response", "reference", "retrieved_contexts")}])
            result = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings, raise_exceptions=True, show_progress=False, batch_size=1)
            score = json.loads(result.to_pandas().to_json(orient="records", force_ascii=False))[0]
            metric_record = {"variant": "agent", "question_id": str(row["question_id"]), "scores": {column: score.get(column) for column in METRIC_COLUMNS}}
            _append_checkpoint(metrics_path, metric_record)
            metric_cache[key] = metric_record
        agent_scores.append({**row, **metric_record["scores"]})
        print(f"Métricas Ragas del agente: {row['question_id']} ({index}/{len(agent_records)})", flush=True)

    import pandas as pd

    baseline_summary = {column: _mean([float(row[column]) for row in baseline_by_id.values() if row.get(column) not in (None, "")]) for column in METRIC_COLUMNS}
    agent_summary = {column: _mean([row.get(column) for row in agent_scores]) for column in METRIC_COLUMNS}
    metric_comparison = [
        {"variant": f"pipeline_top_k_{args.baseline_top_k}", **baseline_summary},
        {"variant": "agent", **agent_summary},
    ]
    _write_csv(output_dir / "metrics_comparison.csv", metric_comparison)
    _write_csv(output_dir / "agent_scores_by_question.csv", agent_scores)

    behavior_rows = []
    if not args.skip_behavior:
        behavior_rows = _run_behavior_cases(Path(args.behavior_cases), output_dir, pipeline, apa_tool, client, args.model, args.only_behavior_case)

    _write_analysis(output_dir / "analysis.md", metric_comparison, cost_rows, behavior_rows)

    pd.DataFrame(agent_scores).to_csv(output_dir / "agent_scores_by_question.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "model": args.model,
        "embedding_model": embedding_name,
        "question_count": len(questions),
        "behavior_case_count": len(behavior_rows),
        "behavior_expectations": "written in evaluation/agent_behavior_cases.json before execution",
        "corpus_sha256": corpus_hash,
        "baseline_ragas_metrics": str(Path(args.baseline_metrics).resolve()),
        "cost_scope": "Groq answer-generation calls only; Ragas judge-token use is excluded from pipeline-vs-agent cost comparison.",
        "note": "The baseline Ragas metrics are reused from the Avance 2 full run. Generation cost/time for both variants is measured in this run.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRagas — RAG fijo frente a agente:")
    print(pd.DataFrame(metric_comparison).set_index("variant").to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nCosto de generación — promedio por pregunta:")
    print(pd.DataFrame(cost_rows).set_index("variant").to_string(float_format=lambda value: f"{value:.2f}"))
    if behavior_rows:
        print(f"\nCasos de comportamiento: {sum(bool(row['passed']) for row in behavior_rows)}/{len(behavior_rows)} cumplidos.")
    print(f"\nResultados guardados en {output_dir}")


if __name__ == "__main__":
    main()
