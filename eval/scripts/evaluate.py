from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.aic_eval import evaluate
from eval.aic_eval.io import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AIC 2026 KIS, Q&A, and TRAKE predictions.")
    parser.add_argument("--labels", required=True, type=Path, help="Ground-truth JSONL file.")
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction JSONL file.")
    parser.add_argument("--summary-out", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--per-query-out", type=Path, help="Optional per-query JSONL output path.")
    args = parser.parse_args()

    result = evaluate(args.labels, args.predictions)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    if args.summary_out:
        write_json(args.summary_out, result["summary"])
    if args.per_query_out:
        write_jsonl(args.per_query_out, result["per_query"])


if __name__ == "__main__":
    main()
