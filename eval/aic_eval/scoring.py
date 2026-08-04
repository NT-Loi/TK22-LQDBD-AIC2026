from __future__ import annotations

from statistics import mean
from typing import Any

from .constants import K_VALUES, TASK_KIS, TASK_QA, TASK_TRAKE
from .normalization import normalize_answer, normalize_video_id, parse_frame_id
from .schema import FrameRange, Label


def in_any_range(frame_id: int | None, ranges: tuple[FrameRange, ...]) -> bool:
    if frame_id is None:
        return False
    return any(start <= frame_id <= end for start, end in ranges)


def answer_video_id(answer: dict[str, Any]) -> str:
    return normalize_video_id(answer.get("video_id") or answer.get("videoId"))


def answer_frame_id(answer: dict[str, Any]) -> int | None:
    for key in ("frame_id", "frameId", "keyframe_index", "keyframe_idx", "frame"):
        if key in answer:
            parsed = parse_frame_id(answer[key])
            if parsed is not None:
                return parsed
    frames = answer.get("display_frames") or answer.get("frames")
    if isinstance(frames, list) and frames:
        first = frames[0]
        if isinstance(first, dict):
            return answer_frame_id(first)
        return parse_frame_id(first)
    return None


def answer_frame_ids(answer: dict[str, Any]) -> list[int | None]:
    raw = answer.get("frame_ids") or answer.get("frameIds") or answer.get("event_frame_ids")
    if isinstance(raw, list):
        return [parse_frame_id(item) for item in raw]

    display_frames = answer.get("display_frames")
    if isinstance(display_frames, list):
        return [answer_frame_id(item) if isinstance(item, dict) else parse_frame_id(item) for item in display_frames]

    frames = answer.get("frames")
    if isinstance(frames, list):
        parsed = [answer_frame_id(item) if isinstance(item, dict) else parse_frame_id(item) for item in frames]
        return [item for item in parsed if item is not None]

    single = answer_frame_id(answer)
    return [single] if single is not None else []


def kis_r_score(answer: dict[str, Any], label: Label) -> float:
    if answer_video_id(answer) != label.video_id:
        return 0.0
    return float(in_any_range(answer_frame_id(answer), label.frame_ranges))


def qa_r_score(answer: dict[str, Any], label: Label) -> float:
    if answer_video_id(answer) != label.video_id:
        return 0.0
    if not in_any_range(answer_frame_id(answer), label.frame_ranges):
        return 0.0
    predicted_answer = normalize_answer(answer.get("answer"))
    return float(predicted_answer in label.answers)


def trake_r_score(answer: dict[str, Any], label: Label) -> float:
    if answer_video_id(answer) != label.video_id:
        return 0.0

    frame_ids = answer_frame_ids(answer)
    if not label.event_frame_ranges:
        return 0.0

    correct = 0
    for frame_id, ranges in zip(frame_ids, label.event_frame_ranges):
        correct += int(in_any_range(frame_id, ranges))
    return correct / len(label.event_frame_ranges)


def r_score(answer: dict[str, Any], label: Label) -> float:
    if label.task == TASK_KIS:
        return kis_r_score(answer, label)
    if label.task == TASK_QA:
        return qa_r_score(answer, label)
    if label.task == TASK_TRAKE:
        return trake_r_score(answer, label)
    raise ValueError(f"unsupported task: {label.task}")


def video_hit(answer: dict[str, Any], label: Label) -> float:
    return float(answer_video_id(answer) == label.video_id)


def max_at(values: list[float], k: int) -> float:
    return max(values[:k]) if values[:k] else 0.0


def first_rank(values: list[float], *, target: float) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value >= target:
            return idx
    return None


def query_metrics(label: Label, answers: list[dict[str, Any]]) -> dict[str, Any]:
    answers = answers[:100]
    r_scores = [r_score(answer, label) for answer in answers]
    video_scores = [video_hit(answer, label) for answer in answers]

    metrics: dict[str, Any] = {
        "query_id": label.query_id,
        "task": label.task,
        "num_answers": len(answers),
    }

    for k in K_VALUES:
        metrics[f"R@{k}"] = max_at(r_scores, k)
        metrics[f"video_recall@{k}"] = max_at(video_scores, k)

    metrics["final_score"] = mean(metrics[f"R@{k}"] for k in K_VALUES)
    first_relevant_rank = first_rank(r_scores, target=1e-12)
    metrics["first_relevant_rank"] = first_relevant_rank
    metrics["mrr"] = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
    return metrics
