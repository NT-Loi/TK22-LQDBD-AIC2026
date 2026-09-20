# Retrieval Architecture & Workflow Diagrams

This document provides visual Mermaid workflows and architecture diagrams for the AIC 2026 Multimodal Video Retrieval Engine.

---

## 1. End-to-End System Architecture

```mermaid
graph TD
    subgraph Input ["User Query Processing"]
        UQ["Vietnamese User Query"] --> QD["Query Decomposition<br/>(Split Compound Sentences into Atomic Clauses)"]
        QD --> QA["Anchor Sub-Query Q_anchor<br/>(Rarest Action / Subject)"]
        QD --> QC["Context Sub-Query Q_context<br/>(Co-occurring Objects / Background)"]
        QD --> OQ["Optional OCR / Audio Query<br/>(Only if explicitly quoted in text)"]
    end

    subgraph Retrieval ["High-Recall Multi-Modal Engine"]
        QA --> QD_SearchA["Qdrant Search: Q_anchor<br/>(SigLIP, SigLIP2, CLIP_H14)"]
        QC --> QD_SearchC["Qdrant Search: Q_context<br/>(SigLIP, SigLIP2, CLIP_H14)"]
        OQ --> ES_Search["Elasticsearch Engine<br/>(PP-OCR & Whisper Transcripts)"]
        QD_SearchA --> NEIGHBOR["Temporal Neighborhood Fusion<br/>Score = S_anchor(f) + γ max S_context(f') for |f'-f| <= W"]
        QD_SearchC --> NEIGHBOR
        ES_Search --> NEIGHBOR
    end

    subgraph Diversification ["Temporal Clustering & Diversification"]
        NEIGHBOR --> CLUST["Group Adjacent Frames per Video<br/>(Cluster shot/temporal duplicates into 1 candidate)"]
    end

    subgraph Verification ["Pixel & Caption Verification (Playbook Rules)"]
        CLUST --> V1["Step 1: Reject Hard Contradictions<br/>(Mismatched color, attire, count, scene)"]
        V1 --> V2["Step 2: Confirm Exact Action & Scene<br/>(Check keyframe & multi-aspect captions)"]
        V2 --> V3["Step 3: Verify Count & Rare Attributes<br/>(Check wearer, accessories, continuous shot)"]
        V3 --> V4["Step 4: Similarity Reranking<br/>(Rank surviving verified candidates)"]
        V4 --> V5["Step 5: Select Peak Keyframe<br/>(Most simultaneous constraints visible)"]
    end

    subgraph Output ["Submission Export & Validation"]
        V5 --> SUB["Generate Submission File<br/>(KIS / QA / TRAKE)"]
        SUB --> VAL["Validator: validate_submission.py<br/>(Strictly increasing, row count <= 100)"]
    end
```

---

## 2. Query Type Workflows

### A. Textual KIS (Known Item Search)

```mermaid
sequenceDiagram
    autonumber
    actor User as Agent / User
    participant Runner as query_runner.py
    participant Ret as RetrievalSystem
    participant Qdrant as Qdrant Vector DB
    participant Cap as Multi-Aspect Captions
    participant Export as Submission CSV

    User->>Runner: Execute query (e.g. "người mặc áo đỏ phát biểu ngoài trời")
    Runner->>Ret: semantic_search(query, models, limit=100)
    Ret->>Qdrant: Query dense vectors (filtered prefix='L')
    Qdrant-->>Ret: Top candidate frames
    Ret-->>Runner: Ranked raw keyframes
    Runner->>Cap: Read data/caption/<video_id>.json for top candidates
    Note over Runner,Cap: Verify [CHỦ THỂ & HÀNH ĐỘNG] and [BỐI CẢNH]
    Runner->>Runner: Deduplicate / cluster nearby frames per video
    Runner->>Export: Write <video_id>, <keyframe_idx> (Max 100 rows)
```

---

### B. TRAKE (Temporal Retrieval and Alignment of Key Events)

```mermaid
sequenceDiagram
    autonumber
    actor User as Agent / User
    participant Runner as query_runner.py
    participant Ret as RetrievalSystem
    participant Qdrant as Qdrant Vector DB
    participant Align as DP Temporal Alignment
    participant Export as Submission CSV

    User->>Runner: Execute sequence [E1, E2, ..., En]
    Note over Runner: Anchor-First: Pin rarest event / end-state shot
    Runner->>Ret: temporal_search(queries=[E1..En], group_by_video=True)
    Ret->>Qdrant: Search each sub-event across keyframes
    Qdrant-->>Ret: Candidate event scores per frame
    Ret->>Align: Dynamic Programming Path Search
    Note over Align: Enforce strictly increasing frame indices: F1 < F2 < ... < Fn
    Align-->>Ret: Optimal monotonic paths per video
    Ret-->>Runner: Ranked video sequences
    Runner->>Export: Write <video_id>, <frame_1>, ..., <frame_n>
```
