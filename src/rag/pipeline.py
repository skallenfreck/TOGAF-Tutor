from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

from src.rag.embeddings import LocalEmbedder
from src.rag.generation import DEFAULT_GROQ_MODEL, TutorAnswer, generate_answer
from src.rag.vector_store import ChromaVectorStore, RetrievedChunk

ABSTENTION_TEXT = (
    "No encontré información suficiente en los documentos cargados para responder "
    "esta pregunta con respaldo."
)

# Each entry is searched as its own short query. Concatenating every synonym into one
# sentence made the embedding too broad and hid exact matches in other PDF extracts.
QUERY_EXPANSIONS = (
    (
        # Only expand when the student explicitly asks for the ADM phase list.
        # A question that merely mentions the ADM cycle (for example requirements
        # management) must not be flooded with unrelated phase descriptions.
        ("fases del adm", "fases adm", "adm phases", "phases of the adm"),
        (
            "ADM phase sequence Preliminary Phase Phase A Architecture Vision "
            "Phase B Business Architecture Phase C Information Systems Architecture "
            "Phase D Technology Architecture Phase E Opportunities and Solutions "
            "Phase F Migration Planning Phase G Implementation Governance "
            "Phase H Architecture Change Management",
            "Architecture Development Method ADM cycle phase list figure",
            "Preliminary Phase objectives establish architecture capability",
            "Phase A Architecture Vision objectives capabilities business value",
            "Phase B Business Architecture objectives target business architecture",
            "Phase C Data Architecture objectives target data architecture",
        ),
    ),
    (
        ("vision de arquitectura", "architecture vision", "fase a"),
        (
            "Phase A Architecture Vision objectives high-level aspirational capabilities business value",
            "Phase A Statement of Architecture Work approval scope stakeholders",
        ),
    ),
    (
        ("arquitectura de negocio", "arquitectura negocio", "business architecture", "negocio"),
        ("Business Architecture Phase B objectives baseline target business capability process organization",),
    ),
    (
        ("arquitectura de datos", "arquitectura datos", "data architecture", "datos"),
        ("Data Architecture Phase C objectives data entities logical data components",),
    ),
    (
        ("arquitectura de aplicaciones", "arquitectura aplicaciones", "application architecture", "aplicaciones"),
        ("Application Architecture Phase C objectives application systems services",),
    ),
    (
        ("arquitectura tecnologica", "technology architecture", "tecnologica", "tecnologia"),
        ("Technology Architecture Phase D objectives infrastructure platforms technology",),
    ),
    (
        ("bloques de construccion", "building blocks", "abb", "sbb"),
        (
            "Architecture Building Blocks ABB",
            "Solution Building Blocks SBB",
            "Data Architecture building blocks Architecture Repository",
        ),
    ),
    (
        ("entregables", "deliverables", "outputs", "productos de trabajo"),
        (
            "Outputs Preliminary Phase architecture capability framework",
            "Outputs Phase A Architecture Vision Statement of Architecture Work",
            "Outputs Phase B Business Architecture deliverables",
            "Outputs Phase C Data Architecture deliverables",
            "ADM outputs deliverables work products by phase",
        ),
    ),
    (
        (
            "gestion de requisitos", "gestión de requisitos", "requirements management",
            "requisitos", "requerimientos",
        ),
        (
            "Requirements Management continuous phase changes to requirements governance processes other phases",
            "Requirements Repository new requirements Statement of Architecture Work",
            "gestión de requisitos fase continua cambios gobernanza otras fases repositorio",
        ),
    ),
)

FOLLOW_UP_MARKERS = (
    "y eso", "y esa", "y ese", "lo anterior", "amplia eso", "profundiza", "continua",
    "dame mas detalle", "explica eso", "que significa eso", "como se relaciona",
    "esa respuesta", "la respuesta anterior", "respuesta anterior", "de lo anterior",
    "que documento respalda", "documento respalda", "entregable debe aprobarse",
    "ese entregable", "ese resultado",
)

_STOP_WORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "como", "con", "cual", "cuales",
    "cuando", "de", "del", "desde", "donde", "el", "ella", "ellas", "ellos", "en", "entre",
    "es", "esa", "ese", "eso", "esta", "este", "esto", "ha", "hay", "la", "las", "lo",
    "los", "mas", "me", "mi", "mis", "no", "o", "para", "pero", "por", "que", "se", "segun",
    "si", "sin", "sobre", "su", "sus", "tambien", "te", "tiene", "un", "una", "unas", "uno",
    "unos", "y", "the", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "this", "to", "what", "which", "with",
}
_GENERIC_DOMAIN_TERMS = {"togaf", "adm", "architecture", "framework", "enterprise", "phase", "method"}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(text))
        if len(token) >= 2 and token not in _STOP_WORDS
    ]


def _asks_for_adm_phase_list(question: str) -> bool:
    normalized = _normalize(question)
    tokens = set(_tokens(question))
    asks_about_phases = bool(tokens.intersection({"fase", "fases", "phase", "phases"}))
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
    return "adm" in tokens and asks_about_phases and asks_for_sequence and not asks_about_requirements


def build_retrieval_queries(
    question: str, history: list[dict[str, str]] | None = None
) -> list[str]:
    """Return a focused Spanish query and separate English terminology queries."""
    current = question.strip()
    normalized = _normalize(current)
    queries = [current]

    if any(marker in normalized for marker in FOLLOW_UP_MARKERS):
        previous_questions = [
            str(item.get("content", "")).strip()
            for item in (history or [])
            if item.get("role") == "user" and str(item.get("content", "")).strip()
        ][-2:]
        previous_answers = [
            str(item.get("content", "")).strip()
            for item in (history or [])
            if item.get("role") == "assistant" and str(item.get("content", "")).strip()
        ][-1:]
        answer_context = previous_answers[-1][:700] if previous_answers else ""
        queries.extend(
            f"{previous} {answer_context} {current}".strip()
            for previous in previous_questions
        )

    if _asks_for_adm_phase_list(current):
        queries.extend(QUERY_EXPANSIONS[0][1])

    for triggers, expansions in QUERY_EXPANSIONS[1:]:
        if any(trigger in normalized for trigger in triggers):
            queries.extend(expansions)

    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = _normalize(query)
        if key and key not in seen:
            unique.append(query)
            seen.add(key)
    return unique


def build_retrieval_query(question: str, history: list[dict[str, str]] | None) -> str:
    """Compatibility helper for logs and callers that expect one printable query."""
    return "\n".join(build_retrieval_queries(question, history))


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, str, str]:
    metadata = chunk.metadata
    return (
        str(metadata.get("source", "")),
        str(metadata.get("page", "")),
        str(metadata.get("chunk_index", "")),
    )


def _bm25_scores(
    chunks: list[RetrievedChunk], queries: list[str]
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
    set[tuple[str, str, str]],
]:
    """Score the small uploaded corpus lexically, including exact TOGAF terms."""
    if not chunks:
        return {}, {}, set()

    tokenized = [_tokens(chunk.text) for chunk in chunks]
    lengths = [len(tokens) for tokens in tokenized]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    total = len(chunks)
    average_length = sum(lengths) / max(1, total)
    bm25_k1 = 1.5
    bm25_b = 0.75
    max_normalized: dict[tuple[str, str, str], float] = defaultdict(float)
    reciprocal_ranks: dict[tuple[str, str, str], float] = defaultdict(float)
    strong_matches: set[tuple[str, str, str]] = set()

    for query in queries:
        query_terms = Counter(_tokens(query))
        if not query_terms:
            continue
        informative_terms = {
            term
            for term in query_terms
            if term in {"abb", "sbb"}
            or (
                term not in _GENERIC_DOMAIN_TERMS
                and math.log(
                    1.0
                    + (total - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                >= 1.2
            )
        }
        # Common footer words such as "TOGAF" appear in nearly every page. They
        # must not turn every chunk into a lexical match by themselves.
        if not informative_terms:
            continue
        scores: list[tuple[int, float]] = []
        for index, terms in enumerate(tokenized):
            frequencies = Counter(terms)
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                doc_frequency = document_frequency[term]
                inverse_frequency = math.log(
                    1.0 + (total - doc_frequency + 0.5) / (doc_frequency + 0.5)
                )
                length_norm = 1.0 - bm25_b + bm25_b * lengths[index] / max(1.0, average_length)
                term_score = inverse_frequency * (
                    frequency * (bm25_k1 + 1.0) / (frequency + bm25_k1 * length_norm)
                )
                score += term_score * min(2.0, 1.0 + math.log(query_frequency))
            if score > 0:
                scores.append((index, score))

        if not scores:
            continue
        max_score = max(score for _, score in scores)
        scores.sort(key=lambda item: item[1], reverse=True)
        for rank, (index, score) in enumerate(scores, start=1):
            key = _chunk_key(chunks[index])
            max_normalized[key] = max(max_normalized[key], score / max_score)
            reciprocal_ranks[key] += 1.0 / (60.0 + rank)
            matched_informative = informative_terms.intersection(tokenized[index])
            if (
                "abb" in matched_informative
                or "sbb" in matched_informative
                or any(
                    math.log(1.0 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)) >= 2.5
                    for term in matched_informative
                )
                or len(matched_informative) >= 2
            ):
                strong_matches.add(key)

    return dict(max_normalized), dict(reciprocal_ranks), strong_matches


def abstention_answer() -> TutorAnswer:
    return TutorAnswer(
        tema="Información no encontrada en el corpus",
        respuesta=ABSTENTION_TEXT,
        ejemplo="",
        nivel="básico",
    )


class RAGPipeline:
    def __init__(
        self,
        store: ChromaVectorStore,
        embedder: LocalEmbedder,
        *,
        min_similarity: float = 0.55,
        model_name: str = DEFAULT_GROQ_MODEL,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.min_similarity = min_similarity
        self.model_name = model_name

    def retrieve(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        corpus_size = self.store.count()
        if corpus_size == 0:
            return []

        queries = build_retrieval_queries(question, history)
        chunks_by_key: dict[tuple[str, str, str], RetrievedChunk] = {}
        vector_reciprocal_ranks: dict[tuple[str, str, str], float] = defaultdict(float)
        best_similarity = 0.0

        # Search the full session index before filtering. Limiting Chroma to top_k first
        # can discard the relevant passage when similar neighboring chunks rank above it.
        for query in queries:
            candidates = self.store.search(query, self.embedder, top_k=corpus_size)
            for rank, chunk in enumerate(candidates, start=1):
                key = _chunk_key(chunk)
                best_similarity = max(best_similarity, chunk.similarity)
                current = chunks_by_key.get(key)
                if current is None or chunk.similarity > current.similarity:
                    chunks_by_key[key] = chunk
                vector_reciprocal_ranks[key] += 1.0 / (60.0 + rank)

        if not chunks_by_key or best_similarity < self.min_similarity:
            return []

        all_chunks = list(chunks_by_key.values())
        lexical_scores, lexical_reciprocal_ranks, strong_lexical_matches = _bm25_scores(
            all_chunks, queries
        )
        denominator = max(1, len(queries)) / 61.0
        normalized_question = _normalize(question)
        asks_for_outputs = any(
            term in normalized_question
            for term in ("entregable", "deliverable", "output", "producto de trabajo")
        )
        asks_for_objectives = any(
            term in normalized_question for term in ("objetivo", "objectives")
        )
        asks_for_phase_list = _asks_for_adm_phase_list(question)
        asks_for_building_blocks = any(
            term in normalized_question for term in ("bloques de construccion", "abb", "sbb")
        )
        asks_for_requirements = any(
            term in normalized_question
            for term in ("gestion de requisitos", "requirements management", "requisitos", "requerimientos")
        )
        domain_terms = {
            "business": ("arquitectura de negocio", "arquitectura negocio", "business architecture", "negocio"),
            "data": ("arquitectura de datos", "arquitectura datos", "data architecture", "datos"),
            "application": ("arquitectura de aplicaciones", "arquitectura aplicaciones", "application architecture", "aplicaciones"),
            "technology": ("arquitectura tecnologica", "arquitectura de tecnologia", "technology architecture", "tecnolog"),
        }
        requested_domains = {
            domain
            for domain, terms in domain_terms.items()
            if any(term in normalized_question for term in terms)
        }
        asks_for_domain_comparison = len(requested_domains) >= 2
        ranked: list[tuple[float, RetrievedChunk]] = []
        for chunk in all_chunks:
            key = _chunk_key(chunk)
            similarity = chunk.similarity
            lexical_score = lexical_scores.get(key, 0.0)
            # Cosine similarity remains the relevance gate. BM25 and reciprocal-rank
            # fusion help exact English terms and Spanish-to-English domain comparisons.
            if similarity < self.min_similarity - 0.08 and key not in strong_lexical_matches:
                continue
            vector_rrf = vector_reciprocal_ranks.get(key, 0.0) / denominator
            lexical_rrf = lexical_reciprocal_ranks.get(key, 0.0) / denominator
            combined_score = (
                0.40 * similarity
                + 0.35 * vector_rrf
                + 0.20 * lexical_score
                + 0.05 * lexical_rrf
            )
            normalized_text = _normalize(chunk.text)
            if asks_for_outputs:
                if any(term in normalized_text for term in ("outputs", "deliverables", "work products")):
                    combined_score += 0.20
                else:
                    combined_score -= 0.20
            if asks_for_objectives and "objectives" in normalized_text:
                combined_score += 0.06
            if asks_for_phase_list and any(
                term in normalized_text
                for term in ("architecture development cycle", "phases of the adm cycle", "adm phases")
            ):
                combined_score += 0.12
            if asks_for_phase_list and "figure 5-1" in normalized_text:
                # The labeled cycle diagram is the corpus' direct evidence for the
                # complete A-H order; keep it ahead of generic phase descriptions.
                combined_score += 1.0
            if asks_for_building_blocks and any(
                term in normalized_text
                for term in ("building blocks", "building block", "abb", "sbb")
            ):
                combined_score += 0.10
            if asks_for_requirements:
                has_requirement_management_evidence = (
                    "requirements management" in normalized_text
                    or "requirements repository" in normalized_text
                    or "changes to requirements" in normalized_text
                    or "requirements are handled" in normalized_text
                    or "continuous phase" in normalized_text
                    or (
                        "requirement" in normalized_text
                        and any(
                            term in normalized_text
                            for term in ("governance", "other phases", "repository")
                        )
                    )
                )
                # Avoid selecting general phase descriptions merely because the
                # question also contains the words ADM or cycle.
                if not has_requirement_management_evidence:
                    continue
                combined_score += 0.24
            if asks_for_domain_comparison:
                evidence_terms = {
                    "business": ("business architecture", "business process", "business domain"),
                    "data": ("data architecture", "data entities", "data domain"),
                    "application": ("application architecture", "application systems", "application domain"),
                    "technology": ("technology architecture", "technology infrastructure", "technology domain"),
                }
                matched_domains = {
                    domain
                    for domain in requested_domains
                    if any(term in normalized_text for term in evidence_terms[domain])
                }
                if not matched_domains:
                    continue
                combined_score += 0.10 * len(matched_domains)
            ranked.append((combined_score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)

        # Prefer corpus-wide coverage for list, output, and multi-domain questions. For a
        # focused question, keep the strongest pages instead of forcing unrelated PDFs in.
        selected: list[RetrievedChunk] = []
        needs_cross_source_coverage = (
            asks_for_phase_list or asks_for_outputs or asks_for_domain_comparison
        )
        if needs_cross_source_coverage:
            seen_sources: set[str] = set()
            for _, chunk in ranked:
                source = str(chunk.metadata.get("source", ""))
                if source not in seen_sources:
                    selected.append(chunk)
                    seen_sources.add(source)
                if len(selected) >= max(1, top_k):
                    return selected[:max(1, top_k)]

        seen_pages = {
            (str(chunk.metadata.get("source", "")), str(chunk.metadata.get("page", "")))
            for chunk in selected
        }
        selected_keys = {_chunk_key(chunk) for chunk in selected}
        for _, chunk in ranked:
            page_key = (str(chunk.metadata.get("source", "")), str(chunk.metadata.get("page", "")))
            key = _chunk_key(chunk)
            if page_key not in seen_pages and key not in selected_keys:
                selected.append(chunk)
                seen_pages.add(page_key)
                selected_keys.add(key)
            if len(selected) >= max(1, top_k):
                return selected[:max(1, top_k)]

        return selected[:max(1, top_k)]

    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        *,
        top_k: int = 5,
        client=None,
    ) -> tuple[TutorAnswer, list[RetrievedChunk]]:
        retrieved = self.retrieve(question, history, top_k=top_k)
        if not retrieved:
            return abstention_answer(), []
        response = generate_answer(
            question,
            retrieved,
            history,
            client=client,
            model_name=self.model_name,
        )
        return response, retrieved
