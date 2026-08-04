from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import SUPPORTED_TASKS, TASK_KIS, TASK_QA, TASK_TRAKE
from .normalization import normalize_answer, normalize_video_id, parse_frame_id


FrameRange = tuple[int, int]


@dataclass(frozen=True)
class Label:
    query_id: str
    task: str
    query: str
    video_id: str
    frame_ranges: tuple[FrameRange, ...] = ()
    answers: tuple[str, ...] = ()
    event_frame_ranges: tuple[tuple[FrameRange, ...], ...] = ()


def parse_ranges(raw: Any, *, context: str) -> tuple[FrameRange, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context}: frame_ranges must be a non-empty list")

    ranges: list[FrameRange] = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError(f"{context}: invalid frame range: {item}")
        start = parse_frame_id(item[0])
        end = parse_frame_id(item[1])
        if start is None or end is None:
            raise ValueError(f"{context}: frame range must contain integers: {item}")
        if start > end:
            start, end = end, start
        ranges.append((start, end))
    return tuple(ranges)


def parse_event_ranges(row: dict[str, Any], *, context: str) -> tuple[tuple[FrameRange, ...], ...]:
    raw_events = row.get("events")
    if isinstance(raw_events, list) and raw_events:
        parsed_events: list[tuple[FrameRange, ...]] = []
        for idx, event in enumerate(raw_events, start=1):
            if isinstance(event, dict):
                ranges = event.get("frame_ranges") or event.get("gt_frame_ranges")
            else:
                ranges = event
            parsed_events.append(parse_ranges(ranges, context=f"{context}: event {idx}"))
        return tuple(parsed_events)

    raw_ranges = row.get("event_frame_ranges") or row.get("gt_event_frame_ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError(f"{context}: TRAKE labels require events or event_frame_ranges")
    return tuple(parse_ranges(ranges, context=f"{context}: event {idx}") for idx, ranges in enumerate(raw_ranges, start=1))


def parse_label(row: dict[str, Any]) -> Label:
    query_id = str(row.get("query_id", "")).strip()
    if not query_id:
        raise ValueError("label row is missing query_id")

    task = str(row.get("task") or row.get("query_type") or "").strip().lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"{query_id}: unsupported or missing task: {task}")

    video_id = normalize_video_id(row.get("video_id") or row.get("gt_video_id"))
    if not video_id:
        raise ValueError(f"{query_id}: label is missing video_id")

    query = str(row.get("query") or row.get("query_text") or "").strip()

    if task == TASK_TRAKE:
        return Label(
            query_id=query_id,
            task=task,
            query=query,
            video_id=video_id,
            event_frame_ranges=parse_event_ranges(row, context=query_id),
        )

    frame_ranges = parse_ranges(row.get("frame_ranges") or row.get("gt_frame_ranges"), context=query_id)
    if task == TASK_KIS:
        return Label(query_id=query_id, task=task, query=query, video_id=video_id, frame_ranges=frame_ranges)

    raw_answers = row.get("answers") or row.get("gt_answers") or row.get("accepted_answers")
    if raw_answers is None and "answer" in row:
        raw_answers = [row["answer"]]
    if not isinstance(raw_answers, list) or not raw_answers:
        raise ValueError(f"{query_id}: Q&A labels require answers")

    answers = tuple(sorted({normalize_answer(answer) for answer in raw_answers if normalize_answer(answer)}))
    if not answers:
        raise ValueError(f"{query_id}: Q&A labels require at least one non-empty answer")

    return Label(
        query_id=query_id,
        task=task,
        query=query,
        video_id=video_id,
        frame_ranges=frame_ranges,
        answers=answers,
    )
