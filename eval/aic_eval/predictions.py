from __future__ import annotations

from typing import Any


def extract_answers(row: dict[str, Any]) -> list[dict[str, Any]]:
    answers = row.get("answers") or row.get("ranked_answers") or row.get("results")
    if isinstance(answers, list):
        return [answer for answer in answers if isinstance(answer, dict)][:100]

    video_ids = row.get("ranked_video_ids")
    frame_ids = row.get("ranked_frame_ids")
    if isinstance(video_ids, list) and isinstance(frame_ids, list):
        return [{"video_id": video_id, "frame_id": frame_id} for video_id, frame_id in zip(video_ids, frame_ids)][:100]

    if "video_id" in row:
        answer: dict[str, Any] = {"video_id": row["video_id"]}
        if "frame_id" in row:
            answer["frame_id"] = row["frame_id"]
        if "frame_ids" in row:
            answer["frame_ids"] = row["frame_ids"]
        if "answer" in row:
            answer["answer"] = row["answer"]
        return [answer]

    return []
