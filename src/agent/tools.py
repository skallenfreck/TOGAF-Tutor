from __future__ import annotations

import base64
import re
import unicodedata
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen
from typing import Any
from pathlib import Path

from pypdf import PdfReader

from src.rag.pipeline import RAGPipeline

MAX_FRAGMENTS_PER_CALL = 5
MAX_CHARS_PER_FRAGMENT = 900
MAX_QUERY_CHARS = 1000


class BuscarEnBaseTool:
    """Busca evidencia citada en el corpus TOGAF local.

    Úsala cuando la pregunta requiera información específica de los documentos
    TOGAF integrados en el tutor. No la uses para saludos, conversación casual ni
    preguntas que no dependan de ese corpus. Devuelve como máximo cinco fragmentos,
    cada uno limitado a 900 caracteres e identificado por documento y página/sección.
    La búsqueda no llama a ningún LLM.
    """

    name = "buscar_en_base"
    description = (
        "Busca evidencia en el corpus TOGAF incluido en el Tutor. Usar para preguntas "
        "sobre TOGAF que requieran respaldo documental. No usar para saludos o charla "
        "casual. Entrada: consulta en lenguaje natural. Devuelve fragmentos citados."
    )

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def invoke(self, input: str | dict[str, Any]) -> dict[str, Any]:
        """Execute a bounded knowledge-base retrieval without invoking an LLM."""
        if isinstance(input, str):
            query = input
        elif isinstance(input, dict):
            query = input.get("consulta", input.get("query", ""))
        else:
            raise TypeError("La entrada debe ser una consulta o un objeto con 'consulta'.")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("La consulta no puede estar vacía.")
        query = query.strip()[:MAX_QUERY_CHARS]

        # El tope se aplica en código para que quien invoque la herramienta no pueda
        # solicitar más contexto del permitido al modelo.
        chunks = self.pipeline.retrieve(query, history=None, top_k=MAX_FRAGMENTS_PER_CALL)
        results = []
        for chunk in chunks[:MAX_FRAGMENTS_PER_CALL]:
            metadata = chunk.metadata
            page = metadata.get("page")
            section = metadata.get("section")
            location = f"página {page}" if page is not None else str(section or "sección no especificada")
            text = chunk.text.strip()
            if len(text) > MAX_CHARS_PER_FRAGMENT:
                text = text[:MAX_CHARS_PER_FRAGMENT].rstrip() + "…"
            results.append(
                {
                    "documento": str(metadata.get("source", "Documento sin nombre")),
                    "ubicacion": location,
                    "pagina": page,
                    "seccion": section,
                    "similitud": round(chunk.similarity, 4),
                    "fragmento": text,
                }
            )

        return {"consulta": query, "cantidad": len(results), "resultados": results}

    def invoke_with_history(
        self, input: str | dict[str, Any], history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """Search using recent dialogue only to resolve follow-up references."""
        query = input if isinstance(input, str) else input.get("consulta", input.get("query", ""))
        if not isinstance(query, str) or not query.strip():
            raise ValueError("La consulta no puede estar vacía.")
        query = query.strip()[:MAX_QUERY_CHARS]
        chunks = self.pipeline.retrieve(query, history=history, top_k=MAX_FRAGMENTS_PER_CALL)
        results = []
        for chunk in chunks[:MAX_FRAGMENTS_PER_CALL]:
            metadata = chunk.metadata
            page = metadata.get("page")
            section = metadata.get("section")
            location = f"página {page}" if page is not None else str(section or "sección no especificada")
            text = chunk.text.strip()
            if len(text) > MAX_CHARS_PER_FRAGMENT:
                text = text[:MAX_CHARS_PER_FRAGMENT].rstrip() + "…"
            results.append({
                "documento": str(metadata.get("source", "Documento sin nombre")),
                "ubicacion": location,
                "pagina": page,
                "seccion": section,
                "similitud": round(chunk.similarity, 4),
                "fragmento": text,
            })
        return {"consulta": query, "cantidad": len(results), "resultados": results}


APA_DOCUMENT_TITLES = {
    "TOGAF_V9.1-Introduccion.pdf": "Introduction to the ADM",
    "TOGAF_V9.1-FasePreliminar.pdf": "Preliminary Phase",
    "TOGAF_V9.1-VisionArquitectura.pdf": "Phase A: Architecture Vision",
    "TOGAF_V9.1-ArquitecturaNegocio.pdf": "Phase B: Business Architecture",
    "TOGAF_V9.1-ArquitecturaInformacion.pdf": "Phase C: Data Architecture",
    "TOGAF_V9.1-ArquitecturaAplicaciones.pdf": "Phase C: Application Architecture",
}


class GenerarReferenciaApaTool:
    """Construye una referencia APA verificable para un extracto TOGAF del corpus.

    Úsala cuando el estudiante solicite la referencia bibliográfica de uno de los
    documentos incluidos. Recibe el nombre del PDF y una página opcional; verifica
    que el archivo exista en el corpus y que la página sea válida. No inventa títulos
    para documentos desconocidos ni necesita llamar a un modelo generativo.
    """

    name = "generar_referencia_apa"
    description = (
        "Genera una referencia APA del extracto TOGAF incluido en el corpus. Usar cuando "
        "el usuario pida una referencia bibliográfica o cita. Argumentos: documento "
        "(nombre exacto o nombre reconocible del PDF) y pagina opcional."
    )

    def __init__(self, corpus_dir: str | Path) -> None:
        self.corpus_dir = Path(corpus_dir)

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        document = str(input.get("documento", "")).strip()
        if not document:
            raise ValueError("Indica el documento del corpus.")
        normalized = re.sub(r"[^a-z0-9]", "", document.casefold())
        if not normalized:
            raise ValueError("El nombre del documento debe contener letras o números.")
        filename = next(
            (name for name in APA_DOCUMENT_TITLES if re.sub(r"[^a-z0-9]", "", name.casefold()) in normalized or normalized in re.sub(r"[^a-z0-9]", "", name.casefold())),
            None,
        )
        if filename is None:
            raise ValueError("Ese documento no pertenece al corpus TOGAF incluido.")
        path = self.corpus_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"No existe el documento del corpus: {filename}")
        reader = PdfReader(str(path))
        page = input.get("pagina")
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError) as exc:
                raise ValueError("La página debe ser un número entero.") from exc
            if page < 1 or page > len(reader.pages):
                raise ValueError(f"La página debe estar entre 1 y {len(reader.pages)}.")
        title = APA_DOCUMENT_TITLES[filename]
        reference = f"The Open Group. (2011). *TOGAF Version 9.1: {title}* [Extracto en PDF]."
        return {
            "documento": filename,
            "pagina": page,
            "referencia_apa": reference,
            "cita_en_texto": f"(The Open Group, 2011{f', p. {page}' if page else ''})",
            "metadatos_verificados": {"autor": "The Open Group", "año": 2011, "obra": "TOGAF Version 9.1"},
        }


MAX_WEB_RESULTS = 3
MAX_WEB_RESULT_CHARS = 500
MAX_WEB_QUERY_CHARS = 300
ALLOWED_WEB_DOMAINS = (
    "opengroup.org",
    "iso.org",
    "nist.gov",
    "gov.co",
    "europa.eu",
)


class BuscarEnWebTool:
    """Busca información pública actual en la web sin enviar datos del corpus.

    Política «primero la base»: solo úsala después de que buscar_en_base no encuentre
    la respuesta y si la pregunta requiere información pública/actual. Recibe únicamente
    la pregunta del estudiante, nunca fragmentos del corpus. Devuelve hasta tres
    resultados breves de dominios oficiales fijados en ALLOWED_WEB_DOMAINS; el LLM no
    puede cambiar esa lista. Requiere conexión a Internet.
    """

    name = "buscar_en_web"
    description = (
        "Solo después de buscar_en_base sin encontrar respuesta, busca información "
        "pública o actual pertinente. La lista fija de sitios oficiales se aplica dentro "
        "de la herramienta. Solo recibe la pregunta del estudiante; nunca pases fragmentos "
        "del corpus."
    )

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        query = str(input.get("consulta_publica", "")).strip()[:MAX_WEB_QUERY_CHARS]
        if not query:
            raise ValueError("La consulta pública no puede estar vacía.")
        from bs4 import BeautifulSoup

        normalized_query = "".join(
            char for char in unicodedata.normalize("NFD", query.casefold())
            if unicodedata.category(char) != "Mn"
        )
        if "togaf" in normalized_query and any(term in normalized_query for term in ("version", "vigente", "reciente", "actual")):
            search_query = "TOGAF version The Open Group certification portfolio"
        elif "togaf" in normalized_query:
            search_query = f"TOGAF {query} The Open Group official"
        else:
            search_query = query
        url = "https://www.bing.com/search?" + urlencode({"q": search_query, "count": "10"})
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urlopen(request, timeout=8) as response:
            html = response.read(1_000_000).decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for row in soup.select("li.b_algo"):
            anchor = row.select_one("h2 a")
            snippet_node = row.select_one(".b_caption p")
            if anchor is None or not anchor.get("href"):
                continue
            result_url = str(anchor["href"])
            parsed = urlparse(result_url)
            if (parsed.hostname or "").casefold().endswith("bing.com"):
                encoded_target = parse_qs(parsed.query).get("u", [""])[0]
                if encoded_target.startswith("a1"):
                    try:
                        padding = "=" * (-len(encoded_target[2:]) % 4)
                        result_url = base64.urlsafe_b64decode(encoded_target[2:] + padding).decode("utf-8", errors="replace")
                    except (ValueError, UnicodeDecodeError):
                        continue
                else:
                    continue
            host = (urlparse(result_url).hostname or "").casefold().removeprefix("www.")
            if not any(host == domain or host.endswith("." + domain) for domain in ALLOWED_WEB_DOMAINS):
                continue
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            results.append({
                "origen": "web",
                "titulo": anchor.get_text(" ", strip=True)[:160],
                "url": result_url,
                "extracto": snippet[:MAX_WEB_RESULT_CHARS],
            })
            if len(results) >= MAX_WEB_RESULTS:
                break
        return {"consulta_publica": query, "cantidad": len(results), "resultados": results}
