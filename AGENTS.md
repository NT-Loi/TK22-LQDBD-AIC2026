# AGENTS.md — Global Agent Rules for TK22-LQDBD-AIC2026

Welcome, Agent. This repository contains the multimodal video retrieval engine and submission pipeline for the **AI Challenge (AIC) 2026**.

You must adhere strictly to the following rules when interacting with this repository and fulfilling user requests:

---

## 1. 🚨 Mandatory Search Methodology Logging
Whenever the user asks you to perform or evaluate a query (Textual KIS, Visual Q&A, or TRAKE):
**You MUST document your exact search methodology in your response.**
Always include a structured section titled:
### `🔍 Search Strategy & Execution Trace`
Containing:
- **Query Decomposition:** How the Vietnamese query was parsed (visual description, OCR text, spoken keywords).
- **Execution & Parameters:** Exact Python call or command, model names (`SigLIP`, `SigLIP2`, `CLIP_H14`), limit, and filters.
- **Visual Inspection Notes (for Q&A & TRAKE):** What visual evidence you directly observed when inspecting keyframe images.
- **Ranking Rationale:** Why the top candidates were selected.

---

## 2. Vietnamese Query Handling Protocol
- Queries received from the competition / user are in **Vietnamese**.
- **🚨 Queries are almost exclusively Pure Event Descriptions:**
  - They describe visual actions, subjects, and scenes (e.g. "người mặc áo đỏ phát biểu ngoài trời").
  - **DO NOT force or hallucinate `ocr_query` or `audio_query` filters** on standard event queries! Doing so will cause **0 results (false negatives)** if the literal words are not printed on screen or spoken.
  - Rely primarily on **Visual Semantic Vector Search** (`SigLIP`, `SigLIP2`, `CLIP_H14`) and **Multi-Aspect Shot Captions** in `data/caption/<video_id>.json`.
  - Only use `ocr_query` or `audio_query` if the query explicitly mentions literal text/quotes (e.g., "có chữ 'ABC'", "người nói câu 'XYZ'").
- **Language Nuance Handling**:
  - **Visual Vector:** SigLIP/SigLIP2 accept Vietnamese directly. Formulate a complementary English translation only when probing CLIP or testing visual embedding ensemble.
  - **Captions:** Multi-aspect captions in `data/caption/<video_id>.json` are in Vietnamese. Read them directly.
  - **OCR Text & Audio Transcript:** If explicitly present, must remain in Vietnamese.
  - **Q&A Answers:** Answer concisely in Vietnamese (e.g. `"màu đỏ"`, `"5 người"`) matching competition conventions.

---

## 3. Skill Reference
- For detailed step-by-step procedures, data paths, code snippets, and submission constraints, consult:
  [skills/aic-video-retrieval/SKILL.md](./skills/aic-video-retrieval/SKILL.md)
- For architecture diagrams and visual workflows, consult:
  [skills/aic-video-retrieval/ARCHITECTURE.md](./skills/aic-video-retrieval/ARCHITECTURE.md)

---

## 4. Video Retrieval Playbook Rules (Essential Checklist)
- **Sub-Clause Decomposition & Temporal Neighborhood Fusion:** DO NOT feed long compound paragraphs (30-50 words) as a single monolithic query! Decompose into 2-3 focused visual sub-clauses (rarest anchor + co-occurring context) and aggregate scores across adjacent frames ($W \approx 30-100$ frames) in the same video segment.
- **Literal Parsing & No Unstated Assumptions:** Never hallucinate posture, camera angle, or location not in query (e.g. "chạm chân" does not imply "đứng chạm chân").
- **Anchor-First for TRAKE / Sequences:** Find the rarest event or distinctive end-state shot first, then verify preceding/succeeding shot boundaries in that video.
- **Candidate Diversification (Temporal Clustering):** Repeated nearby frames from the same video/shot are ONE candidate! Cluster frames so a single video does not consume all Top-5/Top-10 slots.
- **Visible Evidence Decides:** Embeddings propose, visible pixels decide. Never combine an action from one shot with attributes from another cut.
- **Strict Acceptance Order:** 1. Reject hard contradictions $\rightarrow$ 2. Confirm exact action/scene $\rightarrow$ 3. Confirm count/rare attributes $\rightarrow$ 4. Rank by similarity $\rightarrow$ 5. Select peak representative frame.

---

## 5. Execution Tools & Environment
- Always execute Python scripts using the `uv` toolchain:
  ```bash
  uv run python <script_path>
  ```
- Fast CLI search runner:
  ```bash
  uv run python skills/aic-video-retrieval/scripts/query_runner.py --help
  ```
- Submission validator:
  ```bash
  uv run python skills/aic-video-retrieval/scripts/validate_submission.py --help
  ```

