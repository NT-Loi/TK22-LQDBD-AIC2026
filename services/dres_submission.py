"""Validation and payload building for AIC 2026 DRES submissions."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def normalize_video_id(video_id: str) -> str:
    raw = video_id.strip()
    if not raw or "/" in raw or "\\" in raw:
        raise ValueError("Video ID không hợp lệ")
    stem, extension = os.path.splitext(raw)
    normalized = stem if extension.lower() in VIDEO_EXTENSIONS else raw
    if not normalized or not VIDEO_ID_PATTERN.fullmatch(normalized):
        raise ValueError("Video ID không hợp lệ")
    return normalized


def _validate_time_ms(time_ms: int | None) -> int:
    if time_ms is None or isinstance(time_ms, bool) or time_ms < 0:
        raise ValueError("Thời gian submit phải là số nguyên millisecond không âm")
    return int(time_ms)


def _validate_frame_ids(frame_ids: Iterable[int] | None) -> list[int]:
    frames = list(frame_ids or [])
    if not frames:
        raise ValueError("TRAKE cần ít nhất một frame")
    if any(isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 for frame in frames):
        raise ValueError("TRAKE chỉ chấp nhận frame ID nguyên không âm")
    if len(set(frames)) != len(frames):
        raise ValueError("TRAKE không chấp nhận frame trùng nhau")
    if any(current >= following for current, following in zip(frames, frames[1:])):
        raise ValueError("Frame TRAKE phải tăng nghiêm ngặt theo thứ tự thời gian")
    return frames


def build_dres_submission_payload(
    *,
    mode: str,
    video_id: str,
    time_ms: int | None = None,
    answer: str | None = None,
    frame_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    normalized_video_id = normalize_video_id(video_id)
    normalized_mode = mode.strip().lower()

    if normalized_mode == "kis":
        timestamp = _validate_time_ms(time_ms)
        dres_answer: dict[str, Any] = {
            "mediaItemName": normalized_video_id,
            "start": timestamp,
            "end": timestamp,
        }
    elif normalized_mode == "qa":
        timestamp = _validate_time_ms(time_ms)
        normalized_answer = (answer or "").strip()
        if not normalized_answer:
            raise ValueError("Q&A cần câu trả lời")
        if any(character in normalized_answer for character in ("\n", "\r")):
            raise ValueError("Câu trả lời Q&A phải nằm trên một dòng")
        dres_answer = {
            "text": f"QA-{normalized_answer}-{normalized_video_id}-{timestamp}"
        }
    elif normalized_mode == "trake":
        frames = _validate_frame_ids(frame_ids)
        frame_text = ",".join(str(frame) for frame in frames)
        dres_answer = {"text": f"TR-{normalized_video_id}-{frame_text}"}
    else:
        raise ValueError("Loại submission không được hỗ trợ")

    return {"answerSets": [{"answers": [dres_answer]}]}
