# Video Retrieval System

A full-stack, multimodal video retrieval engine supporting zero-shot semantic text-to-video search and temporal multi-event sequence matching.

## Prerequisites & Infrastructure

The project relies on Docker for vector and text search databases.
- **Qdrant**: For fast dense vector similarity search.

### 1. Start Databases
Start the required databases using the provided `docker-compose.yml`:
```bash
docker-compose up -d
```
This will expose Qdrant on `localhost:6333`.

### 2. Data Folder Structure
Ensure your `data/` directory is structured as follows at the root of the project:

```text
data/
├── embedding/              # Pre-extracted embedding .pt files
│   └── <model_name>/       # e.g., CLIP_H14
│       └── keyframe_<idx>.pt
├── keyframe/               # Extracted keyframes (or keyframes/)
│   └── <video_id>/
│       └── keyframe_<idx>.webp
├── object_detection/       # Object detection outputs for ES
├── ocr/                    # OCR outputs for ES
├── shot/                   # Shot boundary mapping JSONs
│   └── all_scenes_<prefix>.json     
└── video/                  # Raw .mp4 video files
```

### 3. Run the Backend Server
Install the required Python dependencies using `uv` (as the project configures dependencies in `pyproject.toml` and locks them in `uv.lock`), and start the server:

```bash
# Install dependencies and sync virtual environment
uv sync

# Start the web interface and API
uv run uvicorn app:app --reload
```

Alternatively, if you do not have `uv` installed, you can use standard `pip` to install the dependencies defined in `pyproject.toml`:

```bash
# Install package dependencies
pip install .

# Start the web interface and API
uvicorn app:app --reload
```
The application will be available at `http://localhost:8000`.

---

## Architecture & Retrieval Logic

The core retrieval engine is built around a dynamic multi-model pipeline. It allows selecting multiple models simultaneously, normalizing their similarity scores, and fusing them together for robust predictions.

### 1. Single Query (Semantic Search)
1. **Encoding**: The user's text query is encoded using the selected CLIP models.
2. **Vector Retrieval**: The engine queries Qdrant to retrieve the `Top K` most visually similar keyframes.
3. **Score Fusion**: If multiple models are selected (e.g., `clip-vit-b-32` and `clip-vit-l-14`), the raw cosine similarities are Min-Max normalized and fused via a weighted average.
4. **Thresholding**: Any frame with a final score below the user-defined `Score Threshold` is discarded.

### 2. Temporal Search (Multi-Query Sequence Matching)
When a user inputs multiple events (e.g., "Event 1: A man runs" -> "Event 2: A car crashes"), the system performs **Temporal Search**.
1. Independent semantic searches are executed for both queries, pulling top candidates for each.
2. Candidates are grouped by `video_id`.
3. A **Depth-First Search (DFS)** algorithm runs to find all valid sequence combinations where:
   - `Event 2` happens strictly after `Event 1`.
   - The gap between `Event 1` and `Event 2` is `<= config.MAX_FRAME_GAP` (default 2000 frames).
4. The valid sequences are ranked by their average score and returned to the UI.

### 3. Group By Shot (Shot-Level Aggregation)
Because standard retrieval matches individual frames, you often get massive combinatorial noise (matching 5 identical frames in Shot A to 5 identical frames in Shot B). Toggling **Group Shots** fixes this by consolidating visual logic:

* **In Single Query**: The backend maps every retrieved frame to its parent `shot_start_frame` and `shot_end_frame` using binary search (`bisect`). It then groups frames belonging to the same shot, calculating a single `avg_score` for the shot.
* **In Temporal Search**: The DFS sequence matcher builds paths out of the *aggregated shots* instead of raw frames. The chronological constraint safely transitions to shot boundaries: `Shot 2` must start *after* `Shot 1` ends, with the distance `<= MAX_FRAME_GAP`. This drastically reduces noise and outputs highly clean, logical sequence matches.
