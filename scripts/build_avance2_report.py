"""Build the Avance 2 PDF from the saved Ragas checkpoints and app captures."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "evaluation" / "results" / "completa_18_principal"
OUT = ROOT / "output" / "pdf" / "Informe_Avance2_Tutor_TOGAF.pdf"
APP_URL = "https://togaf-tutor-gvt5yxbedckatvnvha3xei.streamlit.app/"
REPO_URL = "https://github.com/skallenfreck/TOGAF-Tutor"
CAPTURE_DIR = Path(r"C:\Users\fresn\AppData\Local\Temp")
CAPTURES = [
    (CAPTURE_DIR / "codex-clipboard-0df32d5c-c05a-41ed-b170-728bbae85e24.png", "Consulta sobre la fase A y respuesta generada en la app pública."),
    (CAPTURE_DIR / "codex-clipboard-c923b47b-0cff-446b-b3c0-0c84b75fff9b.png", "Evidencia recuperada: fragmento, documento, página y similitud."),
    (CAPTURE_DIR / "codex-clipboard-7d437ea7-7714-427d-b5ce-875c071779b0.png", "Consulta sobre ADM y respuesta con varias fuentes."),
]

NAVY = colors.HexColor("#132238")
BLUE = colors.HexColor("#1B6CA8")
TEAL = colors.HexColor("#1C8C82")
LIGHT = colors.HexColor("#EEF3F7")
INK = colors.HexColor("#273444")
MUTED = colors.HexColor("#5C6B79")


def register_fonts() -> None:
    regular = Path(r"C:\Windows\Fonts\arial.ttf")
    bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("Arial", str(regular)))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(bold)))


class RagFlow(Flowable):
    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 10.7 * cm

    def draw(self) -> None:
        c = self.canv
        labels = [
            ("Corpus TOGAF 9.1", "PDF autorizados"),
            ("Ingesta local", "PDF / DOCX / TXT / HTML"),
            ("Extracción + chunking", "450 caracteres · solape 80"),
            ("Embeddings FastEmbed", "MiniLM multilingüe · local"),
            ("Índice Chroma", "coseno · fuente y página"),
            ("Recuperación híbrida", "vector + BM25 · top_k 5"),
            ("Generación Groq", "System Prompt + Few-Shot + JSON"),
            ("Respuesta verificable", "evidencia y referencias visibles"),
        ]
        box_w = self.width * 0.68
        box_h = 0.83 * cm
        gap = 0.39 * cm
        x = (self.width - box_w) / 2
        y = self.height - box_h
        for i, (title, detail) in enumerate(labels):
            fill = NAVY if i in (0, 7) else (BLUE if i in (1, 2, 3, 4, 5, 6) else TEAL)
            c.setFillColor(fill)
            c.roundRect(x, y, box_w, box_h, 5, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Arial-Bold", 8.4)
            c.drawCentredString(self.width / 2, y + 0.50 * cm, title)
            c.setFont("Arial", 7.2)
            c.drawCentredString(self.width / 2, y + 0.20 * cm, detail)
            if i < len(labels) - 1:
                c.setStrokeColor(TEAL)
                c.setLineWidth(1.5)
                mid = self.width / 2
                c.line(mid, y - 0.02 * cm, mid, y - gap + 0.10 * cm)
                c.line(mid, y - gap + 0.10 * cm, mid - 0.10 * cm, y - gap + 0.22 * cm)
                c.line(mid, y - gap + 0.10 * cm, mid + 0.10 * cm, y - gap + 0.22 * cm)
            y -= box_h + gap
        # La abstención se activa al no hallar evidencia suficiente.
        c.setFillColor(colors.HexColor("#F2F7F5"))
        c.setStrokeColor(TEAL)
        c.roundRect(self.width * 0.75, self.height - 4.0 * cm, self.width * 0.24, 1.3 * cm, 5, fill=1, stroke=1)
        c.setFillColor(INK)
        c.setFont("Arial-Bold", 7.5)
        c.drawCentredString(self.width * 0.87, self.height - 3.25 * cm, "Sin evidencia")
        c.setFont("Arial", 7)
        c.drawCentredString(self.width * 0.87, self.height - 3.58 * cm, "Abstención")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el checkpoint requerido: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_results() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int | float]], int]:
    all_metrics = read_jsonl(RUN / "metrics_checkpoint.jsonl")
    answers = read_jsonl(RUN / "answers_checkpoint.jsonl")
    dataset = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    answerable_ids = [str(q["id"]) for q in dataset if not q["expects_abstention"]]
    metric_ids = set(answerable_ids[:15])
    metrics = [r for r in all_metrics if str(r["question_id"]) in metric_ids]
    counts_by_variant = defaultdict(int)
    for row in metrics:
        counts_by_variant[row["variant"]] += 1
    if any(counts_by_variant[v] < 15 for v in ("baseline_top_k_3", "candidate_top_k_8")):
        raise RuntimeError(
            "Se necesitan 15 filas por configuración para el comparativo. "
            f"Estado: {dict(counts_by_variant)}; completa las métricas hasta candidate_top_k_8 q15."
        )
    sums: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: defaultdict[str, int] = defaultdict(int)
    for row in metrics:
        variant = row["variant"]
        counts[variant] += 1
        for name, value in row["scores"].items():
            if value is not None:
                sums[variant][name] += float(value)
    means = {variant: {name: total / counts[variant] for name, total in columns.items()} for variant, columns in sums.items()}
    abst: dict[str, dict[str, int | float]] = {}
    for variant in sorted({r["variant"] for r in answers}):
        rows = [r for r in answers if r["variant"] == variant]
        expected = [bool(r["expects_abstention"]) for r in rows]
        did = [bool(r["did_abstain"]) for r in rows]
        abst[variant] = {
            "expected": sum(expected),
            "correct": sum(e and d for e, d in zip(expected, did)),
            "missed": sum(e and not d for e, d in zip(expected, did)),
            "false": sum(not e and d for e, d in zip(expected, did)),
            "chunks": sum(int(r["retrieved_count"]) for r in rows) / max(1, len(rows)),
        }
    return means, abst, len(metric_ids)


def build() -> Path:
    register_fonts()
    means, abstention, n_questions = load_results()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("BodyES", parent=styles["BodyText"], fontName="Arial", fontSize=9.2, leading=13, textColor=INK, spaceAfter=6)
    small = ParagraphStyle("SmallES", parent=body, fontSize=7.6, leading=10, textColor=MUTED)
    h1 = ParagraphStyle("H1ES", parent=styles["Heading1"], fontName="Arial-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=5, spaceAfter=9)
    h2 = ParagraphStyle("H2ES", parent=styles["Heading2"], fontName="Arial-Bold", fontSize=11, leading=14, textColor=BLUE, spaceBefore=7, spaceAfter=5)
    title = ParagraphStyle("TitleES", parent=styles["Title"], fontName="Arial-Bold", fontSize=28, leading=32, textColor=NAVY, alignment=TA_CENTER, spaceAfter=12)
    subtitle = ParagraphStyle("SubES", parent=body, fontSize=13, leading=18, alignment=TA_CENTER, textColor=MUTED)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.7*cm, leftMargin=1.7*cm, topMargin=1.6*cm, bottomMargin=1.5*cm, title="Informe Avance 2 - Tutor TOGAF", author="Equipo Tutor TOGAF")
    story = []
    story += [Spacer(1, 2.2*cm), Paragraph("TUTOR TOGAF", title), Paragraph("Informe de la segunda entrega", subtitle), Spacer(1, 0.7*cm)]
    story += [Paragraph("Asistente académico con recuperación aumentada por generación para consultar extractos autorizados de TOGAF 9.1.", ParagraphStyle("Lead", parent=body, fontSize=12, leading=17, alignment=TA_CENTER)), Spacer(1, 0.5*cm)]
    details = [["Corpus", "6 extractos TOGAF 9.1"], ["Evaluación", f"Ragas · {n_questions} preguntas, 15 respondibles + 3 fuera del corpus"], ["Aplicación pública", APP_URL], ["Repositorio", REPO_URL]]
    tab = Table([[Paragraph(f"<b>{a}</b>", body), Paragraph(b, body)] for a,b in details], colWidths=[3.4*cm, 13.0*cm])
    tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#D4DEE6")),("INNERGRID",(0,0),(-1,-1),0.35,colors.white),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story += [tab, Spacer(1, 1.1*cm), Paragraph("Síntesis", h2), Paragraph("El sistema recibe documentos cargados por el usuario, extrae texto y metadatos, crea fragmentos con embeddings locales, recupera evidencia mediante búsqueda vectorial y léxica y genera respuestas con Groq. Cada respuesta expone documentos, páginas y fragmentos consultados; si falta evidencia, el flujo puede abstenerse.", body), Paragraph("Los PDF fuente están identificados como copias de evaluación y no se distribuyen dentro del repositorio. La aplicación pública solicita que cada usuario cargue documentos que tenga autorización para procesar.", small), PageBreak()]

    story += [Paragraph("Flujo RAG y decisiones", h1), RagFlow(doc.width), Spacer(1, 0.25*cm)]
    decisions = [
        ["Etapa", "Configuración", "Justificación"],
        ["Corpus e ingesta", "6 extractos TOGAF 9.1; PDF, DOCX, TXT y HTML", "Cubre ADM, fase preliminar, visión, negocio y vistas de datos y aplicaciones."],
        ["Chunking", "450 caracteres; solapamiento 80; corte por párrafo/oración/palabra y por página", "Mantiene contexto sin mezclar páginas y conserva la cita de origen."],
        ["Embeddings", "FastEmbed · paraphrase-multilingual-MiniLM-L12-v2", "Multilingüe, local y sin consumo de tokens de API para vectorizar."],
        ["Índice", "Chroma, distancia coseno; metadatos de archivo y página", "Permite persistencia local y trazabilidad visible de fuentes."],
        ["Recuperación", "Vector + BM25; top_k base 3 frente a candidata 8; umbral 0.55", "La búsqueda léxica ayuda con siglas; el experimento mide el efecto de ampliar candidatos."],
        ["Generación", "Groq gpt-oss-20b; System Prompt, Few-Shot, delimitadores y JSON", "Conserva el formato y reglas del Avance 1, limitando el contexto enviado al modelo a fragmentos recuperados."],
    ]
    decision_table = Table([[Paragraph(str(v), small if i else ParagraphStyle("TH", parent=small, fontName="Arial-Bold", textColor=colors.white)) for v in row] for i,row in enumerate(decisions)], colWidths=[2.5*cm, 5.4*cm, 8.5*cm], repeatRows=1)
    decision_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#CBD5DF")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [decision_table, Spacer(1, 0.25*cm), Paragraph("La transcripción manual del diagrama ADM de la introducción se agrega durante la ingesta porque sus rótulos no se extraen como texto del PDF. Esto vuelve recuperables las fases y la gestión de requisitos.", small), PageBreak()]

    story += [Paragraph("Evaluación Ragas y comparación", h1), Paragraph("Se evaluaron 18 preguntas: 15 con respuesta en el corpus y 3 de abstención. Para la iteración se cambió top_k de 3 a 8 y se mantuvieron los demás parámetros. Cada valor es el promedio sobre las mismas preguntas.", body)]
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_labels = {"faithfulness":"Faithfulness", "answer_relevancy":"Answer relevancy", "context_precision":"Context precision", "context_recall":"Context recall"}
    variants = ["baseline_top_k_3", "candidate_top_k_8"]
    if not all(v in means for v in variants):
        raise RuntimeError(f"No se encontraron ambas configuraciones: {list(means)}")
    data = [["Métrica", "Antes · top_k=3", "Después · top_k=8", "Cambio"]]
    for name in metric_names:
        before, after = means[variants[0]][name], means[variants[1]][name]
        data.append([metric_labels[name], f"{before:.3f}", f"{after:.3f}", f"{after-before:+.3f}"])
    mt = Table([[Paragraph(str(v), small if i else ParagraphStyle("MTH", parent=small, fontName="Arial-Bold", textColor=colors.white)) for v in row] for i,row in enumerate(data)], colWidths=[5.8*cm, 3.5*cm, 3.5*cm, 3.4*cm])
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5DF")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story += [mt, Spacer(1, 0.35*cm)]
    lowest = min(metric_names, key=lambda m: means[variants[1]][m])
    interpretations = {
        "faithfulness":"La menor faithfulness sugiere revisar la fidelidad de la generación a la evidencia recuperada y reforzar la instrucción de no añadir afirmaciones sin soporte.",
        "answer_relevancy":"La menor answer relevancy sugiere revisar la formulación de la pregunta/contexto y el prompt de generación para mantener la respuesta centrada en lo solicitado.",
        "context_precision":"La menor context precision apunta a recuperación con fragmentos irrelevantes; conviene ajustar top_k, umbral o reranking para reducir ruido.",
        "context_recall":"La menor context recall apunta a cobertura insuficiente; conviene revisar chunking, consultas equivalentes y estrategia de recuperación.",
    }
    story += [Paragraph("Interpretación", h2), Paragraph(f"La métrica más baja en la configuración candidata fue <b>{metric_labels[lowest]}</b> ({means[variants[1]][lowest]:.3f}). {interpretations[lowest]}", body), Paragraph("El cambio de top_k se considera una iteración experimental, no una mejora automática: la comparación debe leerse junto con las cuatro métricas y el control de abstención.", body)]
    abst_data = [["Control de abstención", "top_k=3", "top_k=8"]]
    for field,label in [("correct","Abstenciones correctas (de 3)"),("missed","Omisiones (de 3)"),("false","Abstenciones falsas (de 15)"),("chunks","Fragmentos promedio recuperados")]:
        vals=[f"{abstention[v][field]:.2f}" if field=="chunks" else str(abstention[v][field]) for v in variants]
        abst_data.append([label,*vals])
    at=Table([[Paragraph(str(v), small if i else ParagraphStyle("ATH", parent=small,fontName="Arial-Bold",textColor=colors.white)) for v in row] for i,row in enumerate(abst_data)],colWidths=[8.3*cm,4.0*cm,4.0*cm])
    at.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),TEAL),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#CBD5DF")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [Spacer(1, 0.15*cm), at, Spacer(1, 0.2*cm), Paragraph("Nota metodológica: Answer relevancy solicitó tres generaciones, pero el proveedor devolvió una en algunas filas; Ragas continuó con esa generación. Esto puede aumentar la variabilidad de esa métrica y se reporta como limitación de la evaluación.", small), PageBreak()]

    story += [Paragraph("Interfaz desplegada y evidencias", h1), Paragraph(f"Aplicación pública: <link href='{APP_URL}' color='#1B6CA8'>{APP_URL}</link><br/>Repositorio: <link href='{REPO_URL}' color='#1B6CA8'>{REPO_URL}</link>", body), Paragraph("La interfaz permite cargar documentos, crear el índice de sesión, conservar el historial, consultar fragmentos y mostrar el documento, la página y la similitud de cada fuente.", body)]
    included = 0
    for path, caption in CAPTURES:
        if not path.exists():
            continue
        img = Image(str(path))
        max_w, max_h = doc.width, 11.7*cm
        scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
        img.drawWidth, img.drawHeight = img.imageWidth*scale, img.imageHeight*scale
        story += [Paragraph(caption, h2), img, Spacer(1, 0.2*cm)]
        included += 1
        if included < len(CAPTURES):
            story.append(PageBreak())
    if included < 2:
        story.append(Paragraph("Se recomienda adjuntar una captura adicional que muestre una pregunta de seguimiento y otra fuera del corpus en la sesión pública antes de entregar.", body))
    story += [Spacer(1,0.15*cm), Paragraph("Privacidad: la aplicación procesa los archivos cargados en la sesión; los embeddings se calculan localmente en el servidor. Para generar respuestas, envía a Groq la pregunta, el historial reciente y los fragmentos recuperados. No se deben publicar claves ni los PDF fuente.", small)]

    def page(canvas, doc_obj):
        canvas.saveState()
        if doc_obj.page > 1:
            canvas.setStrokeColor(colors.HexColor("#D5DEE7"))
            canvas.line(1.7*cm, 1.12*cm, A4[0]-1.7*cm, 1.12*cm)
            canvas.setFont("Arial", 7.5)
            canvas.setFillColor(MUTED)
            canvas.drawString(1.7*cm, 0.78*cm, "Tutor TOGAF · Avance 2")
            canvas.drawRightString(A4[0]-1.7*cm, 0.78*cm, f"Página {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page, onLaterPages=page)
    return OUT


if __name__ == "__main__":
    print(build())
