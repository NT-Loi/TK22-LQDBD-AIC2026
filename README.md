# Video Retrieval System — AIC 2026

A full-stack, multimodal video retrieval engine supporting zero-shot semantic text-to-video search, temporal multi-event sequence matching, metadata filter search (OCR, Audio Transcript, Object Detection).

---

## 🏗️ Prerequisites & Infrastructure

The engine relies on Docker for vector similarity and text search databases:
- **Qdrant** (`:6333`): High-performance vector database for dense feature embeddings and object payload filtering.
- **Elasticsearch** (`:9200`): Fuzzy text search engine for OCR and Whisper transcript matching.

### 1. Start Infrastructure Services
Start the required databases using Docker Compose:
```bash
docker compose up -d
```
Verify that Qdrant (`http://localhost:6333`) and Elasticsearch (`http://localhost:9200`) are running.

---

## 📁 Data Folder Structure

Ensure your `data/` directory is structured as follows at the root of the project:

```text
data/
├── embedding/                  # Pre-extracted visual embedding .pt files
│   ├── FGCLIP2/                # e.g., FGCLIP2 embeddings
│   ├── SigLIP/                 # e.g., SigLIP embeddings
│   └── SigLIP2/                # e.g., SigLIP2 embeddings
├── keyframe/                   # Extracted keyframe WebP images
│   └── <video_id>/             # e.g., L21_V001/
│       └── keyframe_<idx>.webp
├── object_detection/           # YOLOE object detection outputs
│   └── <video_id>/             # e.g., L21_V001/
│       └── keyframe_<idx>.json
├── ocr/                        # OCR outputs per model source
│   ├── PaddleOCR-VL-1.6/       # e.g., PaddleOCR outputs
│   └── PP-OCRv6/               # e.g., PP-OCRv6 outputs
├── transcript/                 # Whisper audio transcript outputs
│   └── <video_id>.json         # Transcribed speech segments per video
├── shot/                       # Shot boundary JSON mappings
│   └── all_scenes_<prefix>.json
├── video/                      # Raw .mp4 video files
│   └── <video_id>.mp4
└── video_metadata.json         # Video FPS and duration metadata
```

---

## 🔄 Re-Ingesting Data (Before Running)

Before running the retrieval system for the first time or after adding new data/embeddings, you **must ingest the data** into Qdrant and Elasticsearch.

### Option 1: Automatic Re-Ingestion via `app.py`
In `app.py`, set `re_ingest=True` in the `lifespan` handler:

```python
# app.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    global system, video_metadata
    # Set re_ingest=True to rebuild Qdrant & Elasticsearch indices on startup
    system = RetrievalSystem(re_ingest=True)
    video_metadata = load_video_metadata()
    yield
```

Then launch the app (see below). Set back to `re_ingest=False` after the first run.

### Option 2: Manual CLI Script Re-Ingestion
To trigger data re-ingestion directly via Python CLI:

```bash
# Re-ingest ALL data (Embeddings, Object Detection, OCR, Transcripts)
uv run python -c "from retrieval_system import RetrievalSystem; RetrievalSystem(re_ingest=True)"
```

Or re-ingest a specific module individually:
```bash
# Ingest Object Detection payload only
uv run python -c "from retrieval_system import RetrievalSystem; sys = RetrievalSystem(); sys.ingest_object_detection()"

# Ingest OCR to Elasticsearch only
uv run python -c "from retrieval_system import RetrievalSystem; sys = RetrievalSystem(); sys.ingest_ocr()"

# Ingest Transcripts to Elasticsearch only
uv run python -c "from retrieval_system import RetrievalSystem; sys = RetrievalSystem(); sys.ingest_transcript()"
```

---

## 🚀 Running the Server

Install Python dependencies using `uv`, generate video metadata, and start the FastAPI web application:

```bash
# 1. Install dependencies
uv sync

# 2. Generate FPS metadata (keyframe -> timestamp mapping)
uv run python utils/video_metadata.py

# 3. Run web application server
uv run uvicorn app:app --reload
```

The web interface will be live at `http://localhost:8000`.
