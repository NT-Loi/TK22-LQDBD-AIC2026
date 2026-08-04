# AIC Evaluation

This folder contains offline evaluators for the three preliminary-round tasks:

- `kis`: Textual Known Item Search
- `qa`: grounded visual Q&A
- `trake`: temporal retrieval and alignment of key events

The evaluator expects JSONL files. Each label row has a `query_id`, a `task`, a target `video_id`, and task-specific ground truth. Each prediction row has the same `query_id` and up to 100 ranked answers.

## Layout

```text
eval/
├── aic_eval/          # reusable scoring package
├── scripts/           # command-line entrypoints
├── datasets/          # labels and dataset notes
└── runs/              # model/system prediction files and score outputs
```

## Label Format

KIS:

```json
{"query_id":"kis_001","task":"kis","query":"Find ...","video_id":"L01_V001","frame_ranges":[[500,510]]}
```

Q&A:

```json
{"query_id":"qa_001","task":"qa","query":"Find ... and answer ...","video_id":"L05_V005","frame_ranges":[[800,900]],"answers":["blue","màu xanh","xanh"]}
```

TRAKE:

```json
{"query_id":"trake_001","task":"trake","query":"Find ...","video_id":"L10_V010","events":[{"name":"take-off","frame_ranges":[[95,105]]},{"name":"landing","frame_ranges":[[195,205]]}]}
```

## Prediction Format

KIS:

```json
{"query_id":"kis_001","answers":[{"video_id":"L01_V001","frame_id":505}]}
```

Q&A:

```json
{"query_id":"qa_001","answers":[{"video_id":"L05_V005","frame_id":888,"answer":"blue"}]}
```

TRAKE:

```json
{"query_id":"trake_001","answers":[{"video_id":"L10_V010","frame_ids":[101,150,203,251]}]}
```

## Run

```bash
python eval/scripts/evaluate.py ^
  --labels eval/datasets/sample/labels.jsonl ^
  --predictions eval/runs/sample_predictions.jsonl ^
  --summary-out eval/runs/sample_summary.json ^
  --per-query-out eval/runs/sample_per_query.jsonl
```

## Main Metrics

For every answer, the evaluator computes task-specific `R-Score`.

- KIS: `1` only when `video_id` matches and `frame_id` is inside a ground-truth frame range.
- Q&A: `1` only when `video_id`, `frame_id`, and normalized answer are all correct.
- TRAKE: `0` when video is wrong; otherwise the fraction of aligned events whose predicted frame is inside the matching event range.

Then it computes:

```text
R@k = max R-Score among the first k answers, for k in {1, 5, 20, 50, 100}
Final Score = mean(R@1, R@5, R@20, R@50, R@100)
```

The summary reports overall metrics and task-level metrics.
