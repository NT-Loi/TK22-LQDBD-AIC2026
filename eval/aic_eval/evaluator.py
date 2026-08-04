from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .constants import K_VALUES, SUPPORTED_TASKS, TASK_KIS, TASK_QA, TASK_TRAKE
from .io import read_jsonl
from .predictions import extract_answers
from .schema import Label, parse_label
from .scoring import query_metrics


def load_labels(path: Path) -> dict[str, Label]:
    labels: dict[str, Label] = {}
    for row in read_jsonl(path):
        label = parse_label(row)
        if label.query_id in labels:
            raise ValueError(f"duplicate label query_id: {label.query_id}")
        labels[label.query_id] = label
    return labels


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError("prediction row is missing query_id")
        if query_id in predictions:
            raise ValueError(f"duplicate prediction query_id: {query_id}")
        predictions[query_id] = extract_answers(row)
    return predictions


def summarize(rows: list[dict[str, Any]], *, prefix: str | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_queries": len(rows),
        "missing_prediction_queries": sum(1 for row in rows if row["num_answers"] == 0),
    }
    metric_keys = [
        "final_score",
        "mrr",
        *(f"R@{k}" for k in K_VALUES),
        *(f"video_recall@{k}" for k in K_VALUES),
    ]
    for key in metric_keys:
        values = [row[key] for row in rows if key in row]
        if values:
            summary[f"mean_{key}"] = mean(values)

    if prefix:
        return {f"{prefix}_{key}": value for key, value in summary.items()}
    return summary


def evaluate(labels_path: Path, predictions_path: Path) -> dict[str, Any]:
    labels = load_labels(labels_path)
    predictions = load_predictions(predictions_path)

    per_query: list[dict[str, Any]] = []
    for query_id, label in labels.items():
        metrics = query_metrics(label, predictions.get(query_id, []))
        per_query.append(
            {
                "query_id": query_id,
                "task": label.task,
                "query": label.query,
                "gt_video_id": label.video_id,
                **metrics,
            }
        )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        by_task[row["task"]].append(row)

    task_summary = {
        task: summarize(by_task.get(task, []))
        for task in (TASK_KIS, TASK_QA, TASK_TRAKE)
        if by_task.get(task)
    }

    summary = {
        "num_labels": len(labels),
        "num_prediction_queries": len(predictions),
        "unknown_prediction_queries": sorted(set(predictions) - set(labels)),
        "overall": summarize(per_query),
        "by_task": task_summary,
    }

    missing_tasks = sorted(SUPPORTED_TASKS - set(by_task))
    if missing_tasks:
        summary["missing_tasks_in_labels"] = missing_tasks

    return {"summary": summary, "per_query": per_query}
