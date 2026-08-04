from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.aic_eval.constants import SUPPORTED_TASKS
from eval.aic_eval.io import write_json, write_jsonl
from eval.aic_eval.latency import DEFAULT_BASE_URL, DEFAULT_MODELS, measure_latency


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(model.strip() for model in value.split(",") if model.strip())
    if not models:
        raise argparse.ArgumentTypeError("models must contain at least one model name")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure /search latency for AIC KIS, Q&A, and TRAKE queries.")
    parser.add_argument("--queries", type=Path, help="Query-only JSONL file used for latency measurement.")
    parser.add_argument("--labels", type=Path, help="Deprecated alias for --queries.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Running app base URL.")
    parser.add_argument("--models", type=parse_models, default=DEFAULT_MODELS, help="Comma-separated models, e.g. SigLIP.")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per query.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs per query, excluded from metrics.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per request in seconds.")
    parser.add_argument("--limit", type=int, default=100, help="Search result limit.")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Search score threshold.")
    parser.add_argument("--group-by-shot", action="store_true", help="Measure search with shot grouping enabled.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between requests in seconds.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(SUPPORTED_TASKS), help="Optional task filter.")
    parser.add_argument("--summary-out", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--per-query-out", type=Path, help="Optional per-query JSONL output path.")
    parser.add_argument("--samples-out", type=Path, help="Optional raw sample JSONL output path.")
    args = parser.parse_args()
    queries_path = args.queries or args.labels
    if not queries_path:
        raise SystemExit("--queries is required")

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be >= 0")

    result = measure_latency(
        queries_path,
        base_url=args.base_url,
        models=args.models,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        timeout=args.timeout,
        limit=args.limit,
        score_threshold=args.score_threshold,
        group_by_shot=args.group_by_shot,
        tasks=set(args.tasks) if args.tasks else None,
        sleep_seconds=args.sleep,
    )

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    if args.summary_out:
        write_json(args.summary_out, result["summary"])
    if args.per_query_out:
        write_jsonl(args.per_query_out, result["per_query"])
    if args.samples_out:
        write_jsonl(args.samples_out, result["samples"])


if __name__ == "__main__":
    main()
