# Tutor TOGAF — Avance 2

Asistente académico para consultar extractos de **TOGAF 9.1** mediante recuperación
aumentada por generación (RAG). El proyecto conserva el System Prompt del Avance 1,
incluye ejemplos Few-Shot y devuelve respuestas JSON con el tema, explicación, ejemplo
y nivel. La interfaz muestra por separado los documentos y páginas recuperados.

## Alcance del corpus

El corpus disponible para esta entrega está compuesto por seis extractos:

- Introducción al ADM.
- Fase preliminar.
- Fase A: Architecture Vision.
- Fase B: Business Architecture.
- Fase C: Data Architecture.
- Fase C: Application Architecture.

Se eligieron estos extractos porque cubren la introducción y preparación del ADM, junto con
las fases A y B y dos vistas de la fase C disponibles para este proyecto. Así el tutor puede
responder preguntas desde el marco general hasta ejemplos de dominios concretos con citas
de página. El corpus no incluye el capítulo de Technology Architecture (fase D) ni el resto
de TOGAF 9.1, por lo que el asistente debe indicar esa limitación cuando corresponda.

La fase D y otros capítulos no están incluidos. El tutor debe reconocer esa limitación
y abstenerse cuando los fragmentos recuperados no permitan responder.

Los PDF suministrados incluyen la marca **“Evaluation Copy. Not for redistribution”**.
Por ese motivo no se guardan en GitHub. Para ejecutar el proyecto, copia tus archivos
autorizados en `data/source_documents/`; en la app desplegada, cada persona carga sus
documentos durante su propia sesión.

## Flujo RAG

```mermaid
flowchart LR
    A[PDF / DOCX / TXT / HTML] --> B[Extracción local]
    B --> C[Fragmentos por página\n450 caracteres + 80 de solapamiento]
    C --> D[FastEmbed\nmodelo multilingüe local]
    D --> E[Chroma\níndice vectorial]
    F[Pregunta + consultas equivalentes separadas] --> G[Embedding local]
    X[Pregunta previa solo si es seguimiento] --> G
    G --> H[Búsqueda vectorial + BM25]
    E --> H
    H --> I{¿Hay evidencia relevante?}
    I -->|No| J[Abstención sin llamada al LLM]
    I -->|Sí| K[System Prompt + fragmentos recuperados]
    K --> L[Groq JSON]
    L --> M[Respuesta + fragmentos, documentos y páginas]
```

### Decisiones técnicas

| Componente | Decisión | Motivo |
| --- | --- | --- |
| Extracción | `pypdf` en modo de diseño + `fontTools`, `python-docx` y BeautifulSoup | El modo de diseño de pypdf conserva mejor los límites entre palabras en estos PDF; `fontTools` permite interpretar fuentes CFF Type1 que aparecen en ellos. |
| Chunking | 450 caracteres, solapamiento de 80; respeta límites de párrafo, oración y palabra, y no cruza páginas de PDF | El modelo de embeddings admite 128 tokens por entrada; fragmentos más cortos reducen el riesgo de que se trunque su contenido al vectorizarlo. El solapamiento conserva contexto en los cortes y cada cita mantiene su página. |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` mediante FastEmbed/ONNX | Modelo multilingüe local para preguntas en español y documentos mayormente en inglés; no requiere llamadas de embeddings a una API. El nombre del modelo y los parámetros de fragmentación quedan guardados en los metadatos de la colección. |
| Índice | Chroma con distancia coseno | Persiste localmente en `data/chroma/`; la app usa una colección temporal por sesión. |
| Recuperación | búsqueda vectorial y léxica BM25, consultas cortas español-inglés, umbral inicial 0.55 y `top_k=5` | Mantiene cada equivalencia separada para no diluir la consulta; BM25 ayuda con nombres y siglas exactas. Busca candidatos antes de limitar los fragmentos que se envían al modelo. |
| Generación | GroqCloud con `openai/gpt-oss-20b` por defecto | Mantiene el System Prompt, Few-Shot, delimitadores y formato JSON del Avance 1; usa Structured Outputs, esfuerzo de razonamiento bajo y un límite de 2048 tokens de salida para dejar espacio al JSON en respuestas extensas. |

El diagrama vectorial del ciclo ADM de la página 4 de `TOGAF_V9.1-Introduccion.pdf`
no produce etiquetas en la extracción de texto. `documents.py` agrega una transcripción
manual de sus rótulos con los mismos metadatos de documento y página para que la secuencia
y la Gestión de Requisitos sean recuperables y citables.

Los parámetros 450/80 y el umbral 0.55 son la configuración inicial documentada. La
evaluación compara `top_k=3` (línea base) con `top_k=8` (candidata), manteniendo iguales
los demás parámetros. La candidata solo debe llamarse “mejora” si los resultados Ragas
lo respaldan.

## Privacidad y servicios externos

- Los embeddings se generan con FastEmbed/ONNX en el proceso que ejecuta la app: en el
  equipo durante el uso local o en el servidor de Streamlit durante el uso público. La
  primera ejecución descarga el modelo desde Hugging Face; los documentos no se cargan allí.
- Al usar la app pública, el navegador envía los archivos cargados al servidor de
  Streamlit para procesarlos en memoria durante la sesión. La aplicación no los guarda
  en el índice persistente del servidor.
- Solo la pregunta, el historial reciente y los fragmentos recuperados se envían a
  Groq para generar una respuesta. Ragas también envía a Groq las preguntas, respuestas,
  referencias y contextos recuperados para calificarlos.
- Groq indica que no retiene por defecto los datos de inferencia; puede conservarlos
  temporalmente hasta 30 días para investigar abuso o resolver problemas de fiabilidad.
  Sus condiciones también indican que no usa entradas y salidas para entrenar modelos
  salvo autorización expresa. Consulta [el manejo de datos](https://console.groq.com/docs/your-data),
  el [acuerdo de servicio](https://console.groq.com/docs/legal/services-agreement) y los
  controles de retención antes de cargar material privado.
- Groq **también tiene límites**: la cuota depende del modelo y la cuenta, e incluye
  solicitudes y tokens por minuto/día. El plan gratuito está limitado; el plan Developer
  ofrece cuotas mayores y cobra por uso. El modelo predeterminado tiene una ventana de
  contexto de 131,072 tokens y un máximo de 65,536 tokens de salida. Consulta los
  [límites vigentes](https://console.groq.com/docs/rate-limits), [modelos y precios](https://console.groq.com/docs/models)
  y [datos](https://console.groq.com/docs/your-data) antes de ejecutar la evaluación completa.
- Nunca agregues `.env`, `secrets.toml`, índices locales ni los PDF fuente a Git.

## Instalación local

Se recomienda Python 3.12.

```powershell
git clone https://github.com/skallenfreck/TOGAF-Tutor.git
cd Tutor-TOGAF
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edita `.env` localmente y asigna `GROQ_API_KEY`. No pegues la clave en GitHub ni en el chat.

Copia los seis archivos TOGAF a `data/source_documents/`. Para crear o actualizar el
índice local persistente:

```powershell
python -m scripts.build_index
```

Ejecuta el modo de consola:

```powershell
python -m src.main
```

La primera carga de embeddings puede descargar aproximadamente 220 MB desde Hugging
Face. El texto y los vectores resultantes se guardan en el equipo, bajo `data/chroma/`.

## Interfaz conversacional

Inicia Streamlit desde la raíz del repositorio:

```powershell
streamlit run app.py
```

En el panel lateral, carga los documentos y selecciona **Crear índice de esta sesión**.
La conversación conserva turnos para resolver preguntas de seguimiento. En cada
respuesta, abre **Evidencia** para revisar cuántos fragmentos y documentos distintos
se usaron, además del archivo, página y texto recuperado. Un fragmento no equivale a
un PDF: varias entradas pueden pertenecer al mismo documento. El índice de la app se
mantiene en memoria durante la sesión.

## Evaluación Ragas

`evaluation/questions.json` contiene 18 preguntas redactadas para los seis extractos:
15 con respuesta en el corpus y 3 sin respuesta disponible. Cada pregunta incluye una
respuesta de referencia y, cuando corresponde, el PDF y la página esperados.

La evaluación ejecuta ambos valores de `top_k` y calcula **faithfulness**,
**answer_relevancy**, **context_precision** y **context_recall**. También genera un
resumen de control de abstenciones para los casos ausentes del corpus.
El tutor y Ragas usan la misma `GROQ_API_KEY`; la evaluación se conecta al endpoint
compatible con OpenAI de Groq para evitar incompatibilidades del adaptador de Ragas.
El script guarda cada respuesta y cada fila de métricas en checkpoints locales dentro
de `evaluation/results/`. Si Groq interrumpe la ejecución por cuota, vuelve a correr el
mismo comando para reanudar; si cambias el corpus, las preguntas, el modelo o los
parámetros, elige una carpeta `--output` nueva.

```powershell
python -m scripts.evaluate_ragas
```

Para una ejecución piloto, puedes limitar el número de preguntas antes de lanzar la
comparación completa:

```powershell
python -m scripts.evaluate_ragas --limit 3
```

La ejecución completa genera solicitudes a Groq tanto para responder las preguntas
como para juzgar las métricas Ragas. Puede consumir parte de la cuota gratuita o generar
un cobro si la cuenta tiene el plan Developer. Los resultados y textos intermedios se
guardan localmente bajo `evaluation/results/`, carpeta excluida de Git. No incluyas
fragmentos del corpus ni archivos intermedios en un repositorio público.

## Despliegue público

1. Confirma que el código esté en `main` del repositorio de GitHub.
2. Entra a [Streamlit Community Cloud](https://share.streamlit.io/) e inicia sesión con
   GitHub.
3. Crea una app desde el repositorio `skallenfreck/TOGAF-Tutor`, rama `main` y archivo
   de entrada `app.py`.
4. Selecciona Python 3.12 y agrega `GROQ_API_KEY` en los secretos de la app; no la
   agregues a un archivo del repositorio.
5. Despliega, verifica que el arranque termine y carga los PDF desde la interfaz.
6. Copia la URL que Streamlit asigne y actualiza la sección siguiente.

**URL pública de la app:** https://togaf-tutor-gvt5yxbedckatvnvha3xei.streamlit.app/

**Repositorio GitHub:** <https://github.com/skallenfreck/TOGAF-Tutor>

## Archivos principales

- `app.py`: interfaz de chat para Streamlit.
- `src/rag/documents.py`: extracción y chunking con metadatos de fuente/página.
- `src/rag/embeddings.py`: embeddings FastEmbed locales.
- `src/rag/vector_store.py`: índice Chroma persistente o temporal.
- `src/rag/pipeline.py`: recuperación, umbral y abstención.
- `src/rag/generation.py`: integración Groq y salida JSON estructurada.
- `prompts/system_prompt.py`: System Prompt y ejemplos Few-Shot del tutor.
- `scripts/build_index.py`: construcción local del índice persistente.
- `scripts/evaluate_ragas.py`: comparación Ragas antes/después.
- `evaluation/questions.json`: 18 preguntas y respuestas de referencia.
- `data/source_documents/README.md`: cómo preparar el corpus privado.
