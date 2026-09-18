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
├── caption/                    # Multi-aspect shot captions (.json)
│   └── <video_id>.json
├── embedding/                  # Pre-extracted visual embedding .pt files
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

## 🎬 Multi-Aspect Shot Captioning (Gemini 2.5 Flash Lite)

A multimodal video captioning pipeline using **Gemini 2.5 Flash Lite** via Vertex AI. For each shot, the engine dynamically samples up to 3 representative keyframes, aligns overlapping Whisper audio transcripts and OCR detections, and generates **7 structured visual aspects**:
1. `[GÓC NHÌN & CỠ CẢNH]`: Camera framing, angles (top-down, low angle, eye-level), and camera motion (static, pan, zoom).
2. `[CHỦ THỂ & HÀNH ĐỘNG]`: Primary subjects, postures (standing, co chân), limb movements, and prop interactions.
3. `[VẬT THỂ & ĐẶC ĐIỂM TRỰC QUAN]`: Literal container colors (bát trắng, chảo đỏ), materials, shapes, and physical state changes (nở phồng, cắt đôi).
4. `[BỐI CẢNH & KHÔNG GIAN]`: Indoor/outdoor environment, lighting, and spatial arrangement.
5. `[CHỮ, LOGO & MÀN HÌNH]`: On-screen text, TV logos, timestamps, lecture slides, and geometric diagrams.
6. `[DIỄN BIẾN THEO THỜI GIAN]`: Chronological progression across the shot (start → middle → end).
7. `[TỔNG THỂ CẢNH QUAY]`: Concise 2–3 sentence natural Vietnamese narrative for semantic vector search.

Outputs are saved to `data/caption/<video_id>.json` with both full markdown text and normalized parsed dictionary keys.

### 1. Configure Environment Variables
Copy `.env.example` to `.env` and set your Google Cloud / Vertex AI credentials:
```bash
cp .env.example .env
```
Ensure your `.env` contains:
```env
LLM_PROVIDER=vertexai
PROJECT_ID=your-gcp-project-id
VERTEX_LOCATION=global
MODEL_ID="gemini-2.5-flash-lite"
```

### 2. Run Caption Generation

#### A. Test Run on a Single Video (e.g. 5 shots only)
```bash
uv run python -m data_processor.caption --video_id L21_V001 --max_shots 5 --overwrite
```

#### B. Process an Entire Video (All Shots)
```bash
uv run python -m data_processor.caption --video_id L21_V001 --concurrency 5
```

#### C. Batch Process by Prefix (e.g. all L21 videos)
```bash
uv run python -m data_processor.caption --prefix L21 --concurrency 5
```

#### D. Batch Process All Videos in Dataset
```bash
uv run python -m data_processor.caption --all --concurrency 5
```

#### E. Smart Resume Capability
The generator automatically tracks already captioned shots. If a batch run is stopped or interrupted:
- Re-running the command automatically detects cached shots and **only processes the remaining uncaptioned shots**.
- Use `--overwrite` if you want to regenerate all shots from scratch.

#### CLI Arguments Reference
| Flag | Type | Description |
| :--- | :--- | :--- |
| `--video_id <ID>` | `str` | Process a single video (e.g. `L21_V001`). |
| `--prefix <PREFIX>` | `str` | Filter video IDs starting with a prefix (e.g. `L21`). |
| `--all` | `flag` | Batch process all videos found in `data/shot/` and `data/keyframe/`. |
| `--max_videos <N>` | `int` | Limit the total number of videos to process. |
| `--max_shots <N>` | `int` | Limit shots per video (convenient for testing). |
| `--concurrency <N>`| `int` | Max parallel API calls (default: `5`). |
| `--overwrite` | `flag` | Re-generate captions even if the video JSON exists. |

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
