from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .constants import SUPPORTED_TASKS, TASK_TRAKE
from .io import read_jsonl


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODELS = ("SigLIP",)


@dataclass(frozen=True)
class LatencyRequest:
    query_id: str
    task: str
    payload: dict[str, Any]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.get("ok")]
    latencies = [float(sample["latency_ms"]) for sample in successful]
    result_counts = [int(sample.get("result_count", 0)) for sample in successful]

    summary: dict[str, Any] = {
        "runs": len(samples),
        "success_count": len(successful),
        "error_count": len(samples) - len(successful),
    }
    if not latencies:
        return summary

    summary.update(
        {
            "mean_ms": round(mean(latencies), 3),
            "min_ms": round(min(latencies), 3),
            "p50_ms": round(percentile(latencies, 0.50) or 0.0, 3),
            "p90_ms": round(percentile(latencies, 0.90) or 0.0, 3),
            "p95_ms": round(percentile(latencies, 0.95) or 0.0, 3),
            "p99_ms": round(percentile(latencies, 0.99) or 0.0, 3),
            "max_ms": round(max(latencies), 3),
            "mean_result_count": round(mean(result_counts), 3) if result_counts else 0.0,
        }
    )
    return summary


def build_latency_request(
    row: dict[str, Any],
    *,
    models: tuple[str, ...],
    limit: int,
    score_threshold: float,
    group_by_shot: bool,
) -> LatencyRequest:
    query_id = str(row.get("query_id") or row.get("id") or "").strip()
    if not query_id:
        raise ValueError("latency query row is missing query_id")

    task = str(row.get("task") or row.get("query_type") or "").strip().lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"{query_id}: unsupported or missing task: {task}")

    text_queries = _build_text_queries(row, task)
    payload = {
        "text_queries": text_queries,
        "anchor_index": 0,
        "models": list(models),
        "objects": [],
        "audio": "",
        "group_by_shot": group_by_shot,
        "score_threshold": score_threshold,
        "limit": limit,
    }
    return LatencyRequest(query_id=query_id, task=task, payload=payload)


def _build_text_queries(row: dict[str, Any], task: str) -> list[str]:
    raw_text_queries = row.get("text_queries")
    if isinstance(raw_text_queries, list):
        text_queries = [str(query).strip() for query in raw_text_queries if str(query).strip()]
        if text_queries:
            return text_queries

    if task == TASK_TRAKE:
        raw_events = row.get("events")
        if isinstance(raw_events, list):
            event_queries: list[str] = []
            for event in raw_events:
                if isinstance(event, dict):
                    text = str(event.get("query") or event.get("name") or "").strip()
                    if text:
                        event_queries.append(text)
                elif isinstance(event, str) and event.strip():
                    event_queries.append(event.strip())
            if len(event_queries) > 1:
                return event_queries

    query = str(row.get("query") or row.get("query_text") or "").strip()
    if not query:
        raise ValueError(f"{row.get('query_id')}: latency query row is missing query or text_queries")
    return [query]


def post_search(base_url: str, payload: dict[str, Any], *, timeout: float) -> tuple[float, int]:
    url = base_url.rstrip("/") + "/search"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw_response = response.read()
    elapsed_ms = (time.perf_counter() - started) * 1000

    try:
        data = json.loads(raw_response.decode("utf-8"))
    except json.JSONDecodeError:
        data = []
    result_count = len(data) if isinstance(data, list) else 0
    return elapsed_ms, result_count


def measure_latency(
    queries_path: Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    models: tuple[str, ...] = DEFAULT_MODELS,
    runs: int = 3,
    warmup_runs: int = 1,
    timeout: float = 120.0,
    limit: int = 100,
    score_threshold: float = 0.3,
    group_by_shot: bool = False,
    tasks: set[str] | None = None,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    rows = read_jsonl(queries_path)
    requests: list[LatencyRequest] = []

    for row in rows:
        request = build_latency_request(
            row,
            models=models,
            limit=limit,
            score_threshold=score_threshold,
            group_by_shot=group_by_shot,
        )
        if tasks and request.task not in tasks:
            continue
        requests.append(request)

    per_query: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []

    for request in requests:
        for _ in range(warmup_runs):
            try:
                post_search(base_url, request.payload, timeout=timeout)
            except Exception:
                pass
            if sleep_seconds:
                time.sleep(sleep_seconds)

        query_samples: list[dict[str, Any]] = []
        for run_idx in range(1, runs + 1):
            sample = {
                "query_id": request.query_id,
                "task": request.task,
                "run": run_idx,
            }
            try:
                latency_ms, result_count = post_search(base_url, request.payload, timeout=timeout)
                sample.update(
                    {
                        "ok": True,
                        "latency_ms": round(latency_ms, 3),
                        "result_count": result_count,
                    }
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                sample.update({"ok": False, "error": str(exc)})
            query_samples.append(sample)
            all_samples.append(sample)
            if sleep_seconds:
                time.sleep(sleep_seconds)

        per_query.append(
            {
                "query_id": request.query_id,
                "task": request.task,
                "text_queries": request.payload["text_queries"],
                "summary": summarize_samples(query_samples),
                "samples": query_samples,
            }
        )

    by_task: dict[str, dict[str, Any]] = {}
    for task in sorted(SUPPORTED_TASKS):
        task_samples = [sample for sample in all_samples if sample["task"] == task]
        if task_samples:
            by_task[task] = summarize_samples(task_samples)

    return {
        "summary": {
            "base_url": base_url,
            "models": list(models),
            "query_count": len(requests),
            "runs_per_query": runs,
            "warmup_runs_per_query": warmup_runs,
            "overall": summarize_samples(all_samples),
            "by_task": by_task,
        },
        "per_query": per_query,
        "samples": all_samples,
    }
