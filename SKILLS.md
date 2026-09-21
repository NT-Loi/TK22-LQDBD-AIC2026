# Available Skills in TK22-LQDBD-AIC2026

This directory contains specialized operational skills designed for Coding Agents (Gemini 3.8 Flash) working on the AIC 2026 video retrieval system.

---

## 1. [aic-video-retrieval](./skills/aic-video-retrieval/SKILL.md)
- **Playbook Guide:** [SKILL.md](./skills/aic-video-retrieval/SKILL.md)
- **Architecture & Workflows:** [ARCHITECTURE.md](./skills/aic-video-retrieval/ARCHITECTURE.md)
- **Playbook Checklist:** [references/playbook.md](./skills/aic-video-retrieval/references/playbook.md)
- **Description:** Complete operational runbook for querying, inspecting keyframes, and generating Top-100 submissions for:
  - **Type 1: Textual KIS** (Known Item Search)
  - **Type 2: Visual Q&A** (Visual Question Answering with Gemini Multimodal Vision)
  - **Type 3: TRAKE** (Temporal Retrieval and Alignment of Key Events)
- **Included Tools:**
  - `query_runner.py`: CLI search runner for quick single-line command querying.
  - `validate_submission.py`: Pre-submission format and integrity checker.
