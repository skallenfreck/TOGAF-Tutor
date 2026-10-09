from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

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

ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
CHECKPOINT_VERSION = 1


def _patch_optional_vertexai_import() -> None:
    """Keep Groq evaluation usable when Ragas imports its optional Vertex AI adapter."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        present = importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        present = False
    if not present:
        # Ragas 0.4.3 imports this optional integration unconditionally, even when
        # llm_factory uses Groq. The Vertex AI types are only used for an isinstance
        # check, so a local placeholder avoids requiring an unrelated provider SDK.
        module = types.ModuleType(module_name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = module


_patch_optional_vertexai_import()
from ragas.embeddings.base import BaseRagasEmbedding


def _load_dataset(path: Path, limit: int | None) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if len(items) < 15:
        raise ValueError("El conjunto de evaluación debe contener al menos 15 preguntas.")
    return items[:limit] if limit else items


def _answer_text(answer) -> str:
    parts = [answer.respuesta.strip()]
    if answer.ejemplo.strip():
        parts.append(f"Ejemplo: {answer.ejemplo.strip()}")
    return "\n".join(part for part in parts if part)


def _looks_like_abstention(text: str) -> bool:
    lowered = text.casefold()
    markers = (
        "no encontré información",
        "no encontré evidencia",
        "no hay información suficiente",
        "no contiene información",
        "no puedo responder",
        "fuera del alcance",
    )
    return any(marker in lowered for marker in markers)


def _run_signature(
    documents_dir: Path,
    questions: list[dict],
    *,
    model: str,
    embedding_model: str,
    threshold: float,
    baseline_top_k: int,
    candidate_top_k: int,
) -> str:
    files = []
    for path in sorted(
        candidate
        for candidate in documents_dir.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
    ):
        files.append(
            {
                "path": path.relative_to(documents_dir).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    try:
        ragas_version = importlib.metadata.version("ragas")
    except Exception:
        ragas_version = "unknown"
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model,
        "ragas_version": ragas_version,
        "embedding_model": embedding_model,
        "threshold": threshold,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_CHUNK_OVERLAP,
        "baseline_top_k": baseline_top_k,
        "candidate_top_k": candidate_top_k,
        "metrics": METRIC_COLUMNS,
        "documents": files,
        "questions": questions,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_checkpoint(output_dir: Path, signature: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "checkpoint_metadata.json"
    answers_path = output_dir / "answers_checkpoint.jsonl"
    scores_path = output_dir / "metrics_checkpoint.jsonl"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            raise SystemExit(
                "Los checkpoints existentes pertenecen a otra configuración o corpus. "
                "Usa una carpeta --output nueva para iniciar otra evaluación."
            )
    else:
        if answers_path.exists() or scores_path.exists():
            raise SystemExit(
                "Hay checkpoints sin metadatos de configuración. Usa una carpeta "
                "--output nueva para evitar mezclar resultados."
            )
        temporary_path = metadata_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps({"signature": signature}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)
    return answers_path, scores_path


def _load_checkpoint(path: Path) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            key = (str(record["variant"]), str(record["question_id"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            # An interrupted final write should not invalidate earlier rows.
            continue
        records[key] = record
    return records


def _append_checkpoint(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class FastEmbedRagasEmbedding(BaseRagasEmbedding):
    """Adapter so Ragas reuses FastEmbed and performs metric embeddings locally."""

    def __init__(self, local_embedder):
        super().__init__()
        self.local_embedder = local_embedder

    def embed_text(self, text: str, **kwargs):
        return self.local_embedder.embed_query(text)

    async def aembed_text(self, text: str, **kwargs):
        return await asyncio.to_thread(self.local_embedder.embed_query, text)

    def embed_texts(self, texts: list[str], **kwargs):
        return self.local_embedder.embed_documents(texts)

    async def aembed_texts(self, texts: list[str], **kwargs):
        return await asyncio.to_thread(self.local_embedder.embed_documents, texts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara RAGAS para top_k=3 y top_k=8 con el corpus TOGAF local."
    )
    parser.add_argument("--documents", default=str(ROOT / "data" / "source_documents"))
    parser.add_argument("--dataset", default=str(ROOT / "evaluation" / "questions.json"))
    parser.add_argument("--output", default=str(ROOT / "evaluation" / "results"))
    parser.add_argument("--limit", type=int, help="Ejecuta solo las primeras preguntas; para un piloto.")
    parser.add_argument("--baseline-top-k", type=int, default=3)
    parser.add_argument("--candidate-top-k", type=int, default=8)
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Falta GROQ_API_KEY. El script requiere Groq para generar respuestas y "
            "calcular las métricas Ragas."
        )

    questions = _load_dataset(Path(args.dataset), args.limit)
    documents_dir = Path(args.documents).resolve()
    source_documents = load_directory(documents_dir)
    chunks = chunk_documents(
        source_documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
    )
    embedding_model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    threshold = float(os.getenv("MIN_RETRIEVAL_SIMILARITY", "0.55"))
    output_dir = Path(args.output)
    signature = _run_signature(
        documents_dir,
        questions,
        model=args.model,
        embedding_model=embedding_model_name,
        threshold=threshold,
        baseline_top_k=args.baseline_top_k,
        candidate_top_k=args.candidate_top_k,
    )
    answers_checkpoint, metrics_checkpoint = _prepare_checkpoint(output_dir, signature)
    embedder = get_local_embedder(embedding_model_name)
    store = ChromaVectorStore(
        path=ROOT / "data" / "chroma" / "evaluation",
        collection_name="togaf_evaluation",
    )
    store.replace_index(
        chunks,
        embedder,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
    )

    from groq import Groq
    from openai import OpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    client = Groq(api_key=api_key)
    # Ragas 0.4.3's Groq adapter is incompatible with some installed Instructor
    # releases. Groq provides an OpenAI-compatible endpoint, so use its official
    # OpenAI client adapter for Ragas while the app itself continues using Groq.
    evaluator_client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory(args.model, provider="openai", client=evaluator_client)
    evaluator_embeddings = FastEmbedRagasEmbedding(embedder)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecisionWithReference(name="context_precision", llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    variants = [
        (f"baseline_top_k_{args.baseline_top_k}", args.baseline_top_k),
        (f"candidate_top_k_{args.candidate_top_k}", args.candidate_top_k),
    ]
    ragas_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    answer_cache = _load_checkpoint(answers_checkpoint)
    expected_answers = len(variants) * len(questions)
    if answer_cache:
        print(
            f"Reanudando: {len(answer_cache)} respuestas ya están guardadas en "
            f"{answers_checkpoint}.",
            flush=True,
        )
    completed_answers = 0
    for variant, top_k in variants:
        pipeline = RAGPipeline(
            store,
            embedder,
            min_similarity=threshold,
            model_name=args.model,
        )
        for question in questions:
            key = (variant, str(question["id"]))
            record = answer_cache.get(key)
            if record is None:
                try:
                    answer, retrieved = pipeline.answer(
                        question["question"],
                        top_k=top_k,
                        client=client,
                    )
                except Exception:
                    print(
                        f"\nLa generación se interrumpió en {variant}, pregunta "
                        f"{question['id']}. Las respuestas ya completadas quedaron "
                        f"guardadas; vuelve a ejecutar el mismo comando para reanudar "
                        f"desde {answers_checkpoint}.",
                        flush=True,
                    )
                    raise
                record = {
                    "variant": variant,
                    "question_id": str(question["id"]),
                    "expects_abstention": bool(question["expects_abstention"]),
                    "did_abstain": _looks_like_abstention(answer.respuesta),
                    "retrieved_count": len(retrieved),
                    "user_input": question["question"],
                    "response": _answer_text(answer),
                    "reference": question["reference"],
                    "retrieved_contexts": [item.text for item in retrieved],
                }
                _append_checkpoint(answers_checkpoint, record)
                answer_cache[key] = record
            ragas_rows.append(record)
            metadata_rows.append(
                {
                    field: record[field]
                    for field in (
                        "variant",
                        "question_id",
                        "expects_abstention",
                        "did_abstain",
                        "retrieved_count",
                    )
                }
            )
            completed_answers += 1
            if completed_answers == len(questions) or completed_answers == expected_answers:
                print(
                    f"Respuestas preparadas: {variant} "
                    f"({completed_answers}/{expected_answers} en esta ejecución)",
                    flush=True,
                )

    import pandas as pd

    metrics_cache = _load_checkpoint(metrics_checkpoint)
    scored_rows = []
    if metrics_cache:
        print(
            f"Reanudando: {len(metrics_cache)} filas de métricas ya están guardadas "
            f"en {metrics_checkpoint}.",
            flush=True,
        )
    for index, (row, metadata) in enumerate(zip(ragas_rows, metadata_rows), start=1):
        key = (str(metadata["variant"]), str(metadata["question_id"]))
        metric_record = metrics_cache.get(key)
        if metric_record is None:
            evaluation_dataset = EvaluationDataset.from_list(
                [
                    {
                        column: row[column]
                        for column in (
                            "user_input",
                            "response",
                            "reference",
                            "retrieved_contexts",
                        )
                    }
                ]
            )
            try:
                result = evaluate(
                    dataset=evaluation_dataset,
                    metrics=metrics,
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    raise_exceptions=True,
                    show_progress=False,
                    batch_size=1,
                )
            except Exception:
                print(
                    f"\nEl cálculo de métricas se interrumpió en {key[0]}, "
                    f"pregunta {key[1]}. Las filas terminadas quedaron guardadas; "
                    f"vuelve a ejecutar el mismo comando para reanudar desde "
                    f"{metrics_checkpoint}.",
                    flush=True,
                )
                raise
            score_row = json.loads(
                result.to_pandas().to_json(orient="records", force_ascii=False)
            )[0]
            metric_record = {
                "variant": key[0],
                "question_id": key[1],
                "scores": {column: score_row.get(column) for column in METRIC_COLUMNS},
            }
            _append_checkpoint(metrics_checkpoint, metric_record)
            metrics_cache[key] = metric_record
        scored_rows.append({**row, **metadata, **metric_record["scores"]})
        print(
            f"Métricas guardadas: {key[0]} {key[1]} ({index}/{len(ragas_rows)})",
            flush=True,
        )

    scores = pd.DataFrame(scored_rows)
    scores.to_csv(output_dir / "scores_by_question.csv", index=False, encoding="utf-8-sig")

    available_metrics = [column for column in METRIC_COLUMNS if column in scores.columns]
    summary = scores.groupby("variant")[available_metrics].mean(numeric_only=True)
    summary.to_csv(output_dir / "metrics_comparison.csv", encoding="utf-8-sig")

    abstention_rows = []
    for variant, frame in scores.groupby("variant"):
        expected = frame["expects_abstention"].astype(bool)
        did_abstain = frame["did_abstain"].astype(bool)
        abstention_rows.append(
            {
                "variant": variant,
                "expected_abstentions": int(expected.sum()),
                "correct_abstentions": int((expected & did_abstain).sum()),
                "missed_abstentions": int((expected & ~did_abstain).sum()),
                "false_abstentions": int((~expected & did_abstain).sum()),
                "mean_retrieved_chunks": float(frame["retrieved_count"].mean()),
            }
        )
    abstention = pd.DataFrame(abstention_rows).set_index("variant")
    abstention.to_csv(output_dir / "abstention_control.csv", encoding="utf-8-sig")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "embedding_model": embedding_model_name,
                "chunk_size_characters": DEFAULT_CHUNK_SIZE,
                "overlap_characters": DEFAULT_CHUNK_OVERLAP,
                "min_similarity": threshold,
                "baseline_top_k": args.baseline_top_k,
                "candidate_top_k": args.candidate_top_k,
                "question_count": len(questions),
                "note": "Raw prompts, retrieved text, and references are not written to Git-tracked files.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nMétricas Ragas (promedio por configuración):")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nControl de abstención:")
    print(abstention.to_string())
    print(f"\nResultados locales guardados en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
