---
name: aic-video-retrieval
description: Comprehensive operational playbook and runbook for Coding Agents (Gemini 3.8 Flash) to execute, inspect, and rank video retrieval queries for the AI Challenge HCMUS 2026 (Preliminary Round). Covers Textual KIS, Visual Q&A, and TRAKE temporal alignment, multimodal query decomposition, RetrievalSystem API usage, and mandatory search trace logging.
---

# AIC 2026 Video Retrieval Skill for Coding Agent

This skill guides a **Coding Agent** (e.g., Gemini 3.8 Flash) through retrieving, visually inspecting, and formatting video search results for the **AI Challenge (AIC) 2026**.

---

## 🚨 MANDATORY REQUIREMENT: Search Trace Logging

**For every query requested by the user, you MUST document your search methodology in your response.**
Always include a clear section titled:
### `🔍 Search Strategy & Execution Trace`
Containing:
1. **Query Decomposition:** How you broke down the natural Vietnamese query into Visual concepts, OCR on-screen text, and Whisper audio keywords.
2. **Search Commands & Parameters:** Exact Python calls, embedding models used (`SigLIP`, `SigLIP2`, `CLIP_H14`), and filter criteria applied.
3. **Visual Inspection Notes (for Q&A & TRAKE):** What specific visual evidence you observed when directly opening the keyframe images with your vision capability.
4. **Ranking & Selection Rationale:** Why the top candidates were chosen and how the candidate list was structured.

---

## 1. System Architecture & Data Reference

### Databases & Services
- **Qdrant Vector Database:** `http://localhost:6333` (Stores dense visual embeddings: SigLIP, SigLIP2, CLIP_H14, plus object detection payloads).
- **Elasticsearch Engine:** `http://localhost:9200` (Stores PaddleOCR / PP-OCR on-screen texts and Whisper audio transcripts).

### File System Layout
All paths are relative to the workspace root:
- **Keyframes (WebP):** `data/keyframe/<video_id>/keyframe_<idx>.webp`
  - Example: `data/keyframe/L21_V001/keyframe_0.webp`
- **Multi-Aspect Shot Captions (JSON):** `data/caption/<video_id>.json`
  - Contains 7 structured aspects in Vietnamese (`[GÓC NHÌN & CỠ CẢNH]`, `[CHỦ THỂ & HÀNH ĐỘNG]`, `[VẬT THỂ & ĐẶC ĐIỂM TRỰC QUAN]`, `[BỐI CẢNH & KHÔNG GIAN]`, `[CHỮ, LOGO & MÀN HÌNH]`, `[DIỄN BIẾN THEO THỜI GIAN]`, `[TỔNG THỂ CẢNH QUAY]`).
- **Video Metadata (FPS & Timestamps):** `data/video_metadata.json`
- **Shot Boundaries:** `data/shot/all_scenes_<prefix>.json`
- **Raw Video Files:** `data/video/<video_id>.mp4`

---

## 2. Query Nature & Multi-Modal Protocol: Pure Visual & Event Descriptions

### ⚠️ CRITICAL REALITY CHECK: Queries are Pure Event Descriptions!
In AIC competitions, queries are almost exclusively **natural visual event descriptions** (actions, subjects, scene contexts, postures).
They **rarely or never contain explicit on-screen text or spoken dialogues**.
- **DO NOT hallucinate or force `ocr_query` or `audio_query` filters** on regular event queries! For example, if the query mentions *"cuộc họp báo"* (press conference) or *"lễ trao giải"* (award ceremony), do **NOT** filter `ocr_query="cuộc họp báo"` — doing so will cause **0 hits (false negatives)** if that literal text is not rendered on screen.
- **`ocr_query` / `audio_query` are strictly OPTIONAL** and should ONLY be used if the query explicitly specifies literal text/quotes (e.g. *"trên bảng có ghi 'ABC'"*, *"người đó nói 'XYZ'"*).

### Protocol for Parsing Pure Visual Queries

| Component | Handling Strategy | Target System |
| :--- | :--- | :--- |
| **Primary Visual Event (Backbone)** | Extract core visual triplets: **[Subject + Action + Scene + Visual attributes]** (e.g., *"diễn giả mặc áo đỏ phát biểu ngoài trời nhiều cây xanh"*).<br>• Use Vietnamese directly for **SigLIP / SigLIP2**.<br>• Formulate a concise English translation for **CLIP_H14** visual alignment. | `system.semantic_search()` (Qdrant) |
| **Multi-Aspect Shot Captions** | Verify retrieved candidates against the 7 visual aspects in `data/caption/<video_id>.json` (`[CHỦ THỂ & HÀNH ĐỘNG]`, `[BỐI CẢNH & KHÔNG GIAN]`, `[VẬT THỂ]`). | Local File System (JSON) |
| **On-screen Text (OCR)** | **Leave `None` by default.** Only populate if query explicitly mentions literal text/logo on screen. | `ocr_query` (Elasticsearch) |
| **Audio Speech Transcript** | **Leave `None` by default.** Only populate if query explicitly mentions literal spoken quotes. | `audio_query` (Elasticsearch) |
| **Q&A Answer Extraction** | Use candidate keyframe images and directly observe with **Gemini Multimodal Vision**. Answer concisely in Vietnamese (e.g. `"màu đỏ"`, `"5"`). | Multimodal Vision |


---

## 3. Video Retrieval Playbook: Golden Rules & SOP Verification

Incorporated from competition-tested best practices ([references/playbook.md](./references/playbook.md)):

### Rule 1: Parse The Query Literally (No Unstated Assumptions)
- **Component Breakdown:** Split into: `scene`, `action`, `participant count`, `objects`, `attributes (colors, wear, accessories)`, `relations`, and `time/sequence`.
- **Hard vs Soft Constraints:** Treat exact rare details as hard constraints: counts, rare colors, hats, glasses, logos, uniforms. Detector bounding boxes and labels are soft evidence only.
- **🚨 Avoid Hallucinated Assumptions:** Never assume posture, camera angle, or location not stated in the query.
  - *Example:* Do not turn *"chạm tay vào ngón chân"* (touching toes) into *"đứng cúi gập người chạm chân"* — a seated or lying position may be the ground truth.
- **Strict-to-Relaxed Strategy:** Search the strict reading first. If insufficient hits occur, systematically relax one constraint at a time and document the relaxation in the trace.

### Rule 2: Retrieve For High Recall & Anchor-First
- **🚨 DO NOT Input Long Monolithic Queries:**
  - Long multi-sentence queries (30-50 words) suffer from severe semantic dilution in vision-language models (CLIP/SigLIP average embeddings across all tokens).
  - Crucially, different sentences in a video description describe actions occurring across **different adjacent frames/shots**!
  - **Protocol:** Decompose complex queries into 2-3 atomic visual sub-clauses:
    - $Q_{anchor}$: The rarest, most distinctive visual action/subject (e.g. *"cặp trâu màu trắng chạy đua"*).
    - $Q_{context}$: Co-occurring scene or background elements (e.g. *"chạy đua trên ruộng bùn"*).
    - Strip narrative fillers like *"Trong clip có cảnh"*, *"Đoạn video ghi lại"*, *"Sau đó xuất hiện"*.
- **Temporal Neighborhood Fusion (Adjacent Frame Co-occurrence):**
  - Search each sub-clause independently or search the anchor sub-clause first.
  - For each candidate video $V$, inspect adjacent keyframes within a temporal window ($W = \pm 30$ to $\pm 100$ frames, ~1-4 seconds, or the same shot).
  - If complementary sub-clauses co-occur across neighboring frames of video $V$, the video's composite score is boosted:
    $$\text{Score}(V, f) = S_{anchor}(f) + \gamma \cdot \max_{|f' - f| \le W} S_{sub}(f')$$
  - Candidates matching only a single generic word in isolation without neighborhood support are demoted.
- **Anchor-First for Sequences & TRAKE:**
  - For multi-shot / multi-event queries, locate the **single rarest anchor event** (or the most distinctive end-state shot) first.
  - Once the candidate video is pinned, inspect preceding and succeeding shot boundaries within that video to align the chronological sub-events.
- **Candidate Diversification (Temporal Clustering):**
  - **CRITICAL:** Repeated nearby frames from the same video/shot are ONE candidate!
  - Cluster candidate frames temporally to prevent a single video from occupying all top-5 or top-10 slots. Allow diverse candidate videos into the top ranks.
- **Prompt Variants / Ensemble:** When a query permits multiple visual descriptions, query SigLIP (Vietnamese) and CLIP (English translation) to form a union of candidates.

### Rule 3: Verify From The Pixels (Ground Truth Inspection)
- **Embeddings propose, visible evidence decides:** Never rely solely on high cosine similarity scores. A high similarity score on the wrong shot of the same video is a common pitfall.
- **Continuous Shot Continuity:** Required frame-level details must coexist in the same continuous shot. **Never combine an action from one shot with attributes from another cut.**
- **Participant Counting:** Count genuine participants, not background pedestrians or bystanders. Occluded people still count when body continuity proves membership.
- **Independent Attribute Verification:** Verify wearer, object, color, and count independently.

### Rule 4: Candidate Acceptance Hierarchy (Strict 5 Steps)
1. **Reject Hard Contradictions:** Immediately eliminate candidates where a required hard attribute (e.g. red shirt, eyeglasses, outdoor setting) is visibly contradicted.
2. **Confirm Exact Action & Scene:** Ensure the core verb and physical environment match.
3. **Confirm Participant Count & Rare Attributes:** Verify the specific number of people and unique objects.
4. **Rank Surviving Candidates:** Use embedding similarity scores and caption overlap only to order the surviving, verified candidates.
5. **Select Peak Frame:** Pick the specific keyframe within the shot where the most constraints are simultaneously and clearly visible.

### Rule 5: Pre-Submission Final Checklist
Before finalizing a submission row or answer:
- [ ] Exact action is visible.
- [ ] Scene and participant group size are plausible.
- [ ] Every stated count, object, color, and wearer relation is confirmed.
- [ ] Evidence belongs to one continuous shot.
- [ ] No hard query detail is contradicted.
- [ ] The submitted keyframe is the clearest, most representative frame of that event.

---

## 4. `RetrievalSystem` API Cheatsheet

Always execute Python code in the project virtual environment via `uv run python`.

### Quick Import & Initialization
```python
from retrieval_system import RetrievalSystem

# Initialize system (re_ingest=False for fast startup)
system = RetrievalSystem(re_ingest=False)
```

### 1. Semantic Search (Textual Visual + OCR + Audio Filters)
```python
results = system.semantic_search(
    query="người phụ nữ mặc áo đỏ phát biểu",   # Visual description
    models=["score"],                          # Or specific: ["SigLIP", "SigLIP2"]
    limit=100,                                 # Number of candidates
    ocr_query="HỘI NGHỊ",                      # Optional: on-screen text
    audio_query="xin kính chào",               # Optional: spoken words
    score_threshold=0.0
)
# Each item in results:
# {
#     "video_id": "L21_V001",
#     "keyframe_idx": 45,
#     "pts_time": 12.5,
#     "score": 0.8234,
#     "model_scores": {...}
# }
```

### 2. Temporal Search (For TRAKE - Multi-event Sequences)
```python
# Event sequence queries in chronological order
queries = [
    {"text": "vận động viên chạy đà"},
    {"text": "vận động viên giậm nhảy rời khỏi mặt đất"},
    {"text": "vận động viên bay qua xà ngang"},
    {"text": "vận động viên rơi xuống đệm"}
]

results = system.temporal_search(
    queries=queries,
    group_by_video=True,    # Must be True for TRAKE to group by video
    limit=100
)
# Each item in results:
# {
#     "video_id": "L10_V010",
#     "path": [frame_idx_1, frame_idx_2, frame_idx_3, frame_idx_4],
#     "scores": [s1, s2, s3, s4],
#     "total_score": 3.42
# }
```

### 3. OCR Search
```python
ocr_results = system.ocr_search(query="BẾN THÀNH", fuzziness="AUTO", size=50)
# Returns list of matched keyframe records with video_id and frame_idx
```

---

## 5. Standard Operating Procedures (SOP) by Query Type

### SOP 1: Textual KIS (Known Item Search)
- **Goal:** Locate the exact video and keyframe for a natural language event description.
- **Output:** Top up to 100 rows of `<video_id>, <frame_id>`.

1. **Deconstruct:** Identify the visual scene, subject actions, background scenery, and any potential on-screen text or audio keywords.
2. **Execute Retrieval:**
   - Run `system.semantic_search(query=visual_desc, ocr_query=ocr_kw, audio_query=audio_kw, limit=100)`.
3. **Verify Context:**
   - For top 5 candidate videos, read the multi-aspect caption from `data/caption/<video_id>.json` around `keyframe_idx` to confirm visual alignment (`[CHỦ THỂ & HÀNH ĐỘNG]`, `[BỐI CẢNH]`).
4. **Format & Export:**
   - Save candidate pairs `video_id, keyframe_idx` in descending order of `score`.

---

### SOP 2: Visual Q&A (Question Answering)
- **Goal:** Find the keyframe that shows the event AND answer the specific question about it.
- **Output:** Top up to 100 rows of `<video_id>, <frame_id>, <answer>`.

1. **Split Description & Question:**
   - Description part $\rightarrow$ used to retrieve the relevant scene.
   - Question part $\rightarrow$ target information to extract (e.g., color, quantity, text, location).
2. **Retrieve Candidate Keyframes:**
   - Run `system.semantic_search(query=event_description, limit=20)`.
3. **Multimodal Vision Inspection (Agent Core Superpower):**
   - Take the top keyframe candidate paths: `data/keyframe/<video_id>/keyframe_<idx>.webp`.
   - Directly view the image using your multimodal capability (`view_file` or image inspection).
   - Answer the question based on direct visual evidence (e.g., count the people, inspect garment color, check object in hand).
4. **Format Answer:**
   - Keep answers clean and concise in Vietnamese (e.g. `"màu đỏ"`, `"5"`, `"bên trái"`).
5. **Format & Export:**
   - Save `video_id, keyframe_idx, answer` in descending order of confidence.

---

### SOP 3: TRAKE (Temporal Retrieval and Alignment of Key Events)
- **Goal:** Find a single video containing a sequential chain of $N$ sub-events and output one semantic keyframe for each event.
- **Output:** Top up to 100 rows of `<video_id>, <frame_1>, <frame_2>, ..., <frame_n>`.
- **CRITICAL CONSTRAINT:** Within each row, frame indices **MUST be strictly increasing**: $frame\_1 < frame\_2 < \dots < frame\_n$.

1. **Deconstruct Sequence:**
   - Break query into ordered events: $E_1, E_2, \dots, E_N$.
2. **Execute Temporal Search:**
   - Run `system.temporal_search(queries=[{"text": e} for e in events], group_by_video=True, limit=100)`.
3. **Keyframe Alignment & Fine-tuning:**
   - For the highest scoring candidate video, verify the frame timestamps using `data/video_metadata.json` or inspect nearby keyframe images to ensure the exact semantic moment is captured (e.g. take-off instant).
   - Verify that $frame_1 < frame_2 < \dots < frame_n$.
4. **Format & Export:**
   - Save `video_id, frame_1, frame_2, ..., frame_n`.

---

## 6. Helper CLI Scripts

Two ready-to-use scripts are provided in `skills/aic-video-retrieval/scripts/`:

### 1. `query_runner.py`
Run quick searches directly from the terminal:
```bash
# Textual KIS
uv run python skills/aic-video-retrieval/scripts/query_runner.py --type kis --query "diễn giả mặc áo đỏ phát biểu ngoài trời" --limit 100 --output submission_kis.csv

# Visual Q&A
uv run python skills/aic-video-retrieval/scripts/query_runner.py --type qa --query "lễ trao giải thưởng âm nhạc" --limit 20 --output candidates_qa.json

# TRAKE
uv run python skills/aic-video-retrieval/scripts/query_runner.py --type trake --events "chạy đà" "giậm nhảy" "bay qua xà" "tiếp đất" --limit 100 --output submission_trake.csv
```

### 2. `validate_submission.py`
Validate your submission file against competition rules before delivery:
```bash
uv run python skills/aic-video-retrieval/scripts/validate_submission.py --file submission_kis.csv --type kis
uv run python skills/aic-video-retrieval/scripts/validate_submission.py --file submission_trake.csv --type trake
```

---

## 7. Response Template for the Agent

When answering the user's retrieval request, structure your response as follows:

```markdown
### 🔍 Search Strategy & Execution Trace
- **Query Decomposition:**
  - Visual: `<Visual concepts>`
  - OCR Text: `<On-screen keywords or None>`
  - Audio/Transcript: `<Spoken keywords or None>`
- **Execution:** `<Script / command / parameters used>`
- **Visual Inspection (if QA/TRAKE):** `<What was observed in keyframe images>`
- **Rationale:** `<Why top candidates were selected>`

### 📊 Top Results Summary
| Rank | Video ID | Keyframe ID | Score / Answer | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 1    | L21_V001 | 1500        | 0.892 / "Màu đỏ"| Confirmed via keyframe view |
...

### 📁 Submission File
Submission saved to `submission/<filename>.csv` (Validated: PASSED ✅).
```
