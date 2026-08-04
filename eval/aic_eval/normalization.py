from __future__ import annotations

import re
import unicodedata
from typing import Any


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "khong": "0",
    "mot": "1",
    "hai": "2",
    "ba": "3",
    "bon": "4",
    "tu": "4",
    "nam": "5",
    "sau": "6",
    "bay": "7",
    "tam": "8",
    "chin": "9",
    "muoi": "10",
}


def normalize_video_id(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    return text[:-4] if text.lower().endswith(".mp4") else text


def parse_frame_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d+", text):
            return int(text)
    return None


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = strip_accents(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return NUMBER_WORDS.get(text, text)
