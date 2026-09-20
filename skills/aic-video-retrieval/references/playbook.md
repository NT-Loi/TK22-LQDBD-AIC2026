# Video Retrieval Playbook

Use this checklist for KIS and video-retrieval queries.

## 1. Parse The Query Literally

- Split it into: scene, action, people count, objects, attributes, relations, and time.
- Treat exact rare details as hard constraints: counts, colors, hats, glasses, signs, or identities.
- Never add an unstated assumption such as posture, camera angle, or location.
- If wording is ambiguous, search the strict reading first and label every relaxation.

## 2. Retrieve For High Recall

- **Decompose compound queries:** Never feed a long multi-sentence paragraph as one monolithic query. Split into 2-3 atomic visual sub-clauses (rarest anchor + co-occurring context).
- **Temporal Neighborhood Aggregation:** Check adjacent frames in the same shot/window ($W \approx 30-100$ frames) for co-occurrence of complementary sub-clauses.
- For speed, start with the rarest searchable clue in ASR, OCR, or captions; use embeddings first only when text evidence is weak.
- For multi-shot queries, locate one rare anchor, then verify the described shot order inside the same video.
- For templated sequences, retrieve the most distinctive end-state shot first, then inspect only its preceding shot boundaries.
- Run semantic search and structured search independently, then take their union.
- Search with a few faithful prompt variants when the query permits multiple visual forms.
- Use detector counts and labels as soft evidence, never as hard gates.
- Diversify candidates by video and temporal cluster; repeated nearby frames are one candidate.
- If filtering may have hidden the answer, audit the full embedding set without metadata gates.

## 3. Verify From The Pixels

- Embeddings, detectors, OCR, ASR, and captions propose candidates; visible evidence decides.
- Inspect the original-resolution frame and a dense sequence across its full shot.
- Required frame-level details must coexist in the same frame or continuous shot.
- Count participants, not every detected person. Occluded people still count when body continuity proves membership.
- Verify each rare attribute separately: wearer, object, color, and exact count.
- Bounding boxes and color masks help rank candidates but require manual confirmation.
- Never combine an action from one shot with attributes from another cut.
- When candidates share the same objects, compare camera direction and transition order; content labels alone cannot identify the edit.

## 4. Accept Candidates In This Order

1. Reject any hard contradiction.
2. Confirm the exact action and scene.
3. Confirm participant count and rare attributes.
4. Use similarity scores only to rank the surviving candidates.
5. Select the frame where the most constraints are simultaneously visible.

If a hard constraint remains uncertain, keep searching or report the uncertainty explicitly.

## 5. Failure Lessons

- Do not turn "touching toes" into "standing toe touch"; a seated version may be correct.
- Hard `person >= N` filters fail under occlusion, partial bodies, and detector misses.
- Hat and glasses labels can confuse hair, goggles, or headwear.
- Red-color masks can count clothing, backgrounds, or small trim on a non-red hat.
- A strong video match can still point to the wrong frame; inspect the whole shot.
- Many near-duplicate frames from one video create false confidence.
- Whenever possible, confirm the final candidate through a second independent retrieval path.

## Final Check

- Exact action is visible.
- Scene and group size are plausible.
- Every stated count, object, color, and wearer relation is verified.
- Evidence belongs to one continuous shot.
- No hard query detail is contradicted.
- The submitted frame is the clearest representative of that shot.
