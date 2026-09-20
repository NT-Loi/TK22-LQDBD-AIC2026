---
name: aic-frame-retrieval
description: >
  Retrieve AIC video frames from natural-language prompts describing a sequence
  of events. The Coding Agent is the sole planner and visual verifier.
  Strategy: extract the most distinctive sub-event (Special Detail), search it
  first with a clear English query, deduplicate results by video/time, render a
  contact sheet, then expand temporally (backward/forward/both) to verify the
  full event sequence. Output is a Suspects List — always show video_id +
  frame_idx. Do not submit without explicit user confirmation.
---

# AIC Frame Retrieval — Special Detail + Temporal Expansion

The Coding Agent is the LLM core. No second model is used.

---

## BƯỚC 0 — Parse & Identify Special Detail

Decompose the user prompt into numbered events **[1], [2], ..., [N]**.

Then select the **Special Detail**: the single sub-event that is most visually
distinctive / rare / specific.

**Criteria (priority order):**
1. Combines multiple unusual elements in one frame (e.g., two devices in two hands)
2. Very specific action or object (not an everyday scene)
3. Has quantitative detail (count, specific color, named item)

For the Special Detail, determine:
- **Position K/N** — "event K out of N total events"
- **Expansion direction:**
  - K = 1        → FORWARD
  - 1 < K < N    → BIDIRECTIONAL
  - K = N        → BACKWARD
- **en_query** — full English description of only that sub-event.
  No length limit. Focus on nouns, verbs, and visual objects.
  Omit generic words ("city street", "background").

---

## BƯỚC 1 — Anchor Search (one search, one deduplication)

```bash
.venv/bin/python3 .agents/skills/aic-frame-retrieval/scripts/retrieval_agent.py search \
  --text "<en_query>" \
  --top 20 \
  --min-gap 180 \
  --output artifacts/aic_agent/anchor.json \
  --position "K/N" \
  --direction "BACKWARD|BIDIRECTIONAL|FORWARD"
```

**Deduplication rule:** same video_id → only kept if frames are ≥ 3 minutes apart.

Then render the contact sheet immediately:

```bash
.venv/bin/python3 .agents/skills/aic-frame-retrieval/scripts/retrieval_agent.py sheet \
  --input artifacts/aic_agent/anchor.json \
  --limit 20 --columns 5
```

**After viewing the sheet:**  
Mark each frame as SUSPECT (looks like the special detail, even approximately) or
REJECT (clearly wrong scene/action). Candidates do NOT need to match 100% — roughly
similar is enough at this stage.

---

## BƯỚC 2 — Temporal Expansion

For each SUSPECT anchor frame (video_id=V, keyframe_index=F):

```bash
.venv/bin/python3 .agents/skills/aic-frame-retrieval/scripts/retrieval_agent.py context \
  --video-id V --keyframe-index F \
  --neighbors 12 --columns 4
```

**What to look for (direction-dependent):**
- BACKWARD  → look at frames *before* F for events [1]..[K-1]
- FORWARD   → look at frames *after*  F for events [K+1]..[N]
- BIDIRECTIONAL → both

**Matching surrounding events: approximate is fine.**  
If the frame "looks like" the described scene — include it in suspects.

---

## BƯỚC 3 — Output: Suspects List

The agent always prints the **full top-20 anchor list** plus any suspects found
from temporal expansion, in this format:

```
════════════════════════════════════════════════════
ANCHOR SEARCH KẾT THÚC
Query đã dùng: "..."
Special detail: event [K/N]
Hướng mở rộng: BACKWARD | BIDIRECTIONAL | FORWARD
════════════════════════════════════════════════════

📋 TOP ANCHOR FRAMES:
  #    video_id      frame_idx      time      score
  1    L23_V018           1056   00:42.2     0.3120
  2    L23_V014           7338   04:53.5     0.2960
  ...

📋 SUSPECTS LIST (sau temporal expansion):
  #    video_id      frame_idx      note
  1    L23_V018           600    ← event [1] (trước anchor)
  2    L23_V018           800    ← event [2] (trước anchor)
  3    L23_V018          1056    ← ANCHOR (event [3/3])
  ...

ℹ️  Query đã dùng: "..."
    Để tìm lại với query khác → xem Options bên dưới.
════════════════════════════════════════════════════
```

---

## Options: Re-search

**Option A** — Chọn Special Detail khác (event khác trong prompt)
**Option B** — Sửa en_query (thêm/bớt từ, đổi góc mô tả)
**Option C** — Thêm OCR filter: `--ocr "text trên màn hình"`
**Option D** — Thêm Audio filter: `--audio "lời thoại"`

Each re-search is independent — always print a full new Suspects List.

---

## Stopping Conditions

- Stop after **2 re-search rounds** if no convincing suspect found.
- Report which constraints could not be verified and suggest the next best query.
- **Never fabricate** a match. If an image is missing, mark it `metadata-only`.
- **Never submit** via `/api/submit` unless user explicitly confirms.

---

## Quick Reference

```bash
# Health check
.venv/bin/python3 .agents/skills/aic-frame-retrieval/scripts/retrieval_agent.py audit

# Search (main anchor search)
.venv/bin/python3 .agents/skills/aic-frame-retrieval/scripts/retrieval_agent.py search \
  --text "..." [--ocr "..."] [--audio "..."] \
  --top 20 --min-gap 180 --output anchor.json

# Sheet
.venv/bin/python3 .agents/skills/aic-frame-retrieval/scripts/retrieval_agent.py sheet \
  --input anchor.json --limit 20 --columns 5

# Context (temporal expansion)
.venv/bin/python3 .agents/skills/aic-frame-retrieval/scripts/retrieval_agent.py context \
  --video-id L23_V018 --keyframe-index 1056 --neighbors 12

# All keyframes for a video
.venv/bin/python3 .agents/skills/aic-frame-retrieval/scripts/retrieval_agent.py keyframes \
  --video-id L23_V018
```

