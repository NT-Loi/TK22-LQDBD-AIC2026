"""Resume-safe ingestion of one named vision vector into an existing Qdrant collection.

Unlike ``RetrievalSystem.ingest_vision_embedding``, this utility preserves the other
named vectors already stored on a point. Existing points are updated with
``update_vectors``; missing points are inserted with ``upsert``.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import OptimizersConfigDiff, PointStruct, PointVectors
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATA_DIR,
    QDRANT_COLLECTION_NAME,
    QDRANT_HOST_URL,
    VISION_EMBEDDING_DIM,
)


def _point_id(video_id: str, frame_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{frame_idx}"))


def _load_vector(path: Path, expected_dim: int) -> list[float]:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    vector = tensor.detach().float().reshape(-1)
    if vector.numel() != expected_dim:
        raise ValueError(f"{path}: expected {expected_dim} values, got {vector.numel()}")
    return vector.numpy().tolist()


def ingest(
    model: str,
    batch_size: int,
    loader_workers: int,
    checkpoint: Path,
    wait: bool = True,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict:
    embedding_root = Path(DATA_DIR) / "embedding" / model
    keyframe_root = Path(DATA_DIR) / "keyframe"
    expected_dim = VISION_EMBEDDING_DIM[model]
    client = QdrantClient(url=QDRANT_HOST_URL, prefer_grpc=True, timeout=180)

    completed = set()
    if checkpoint.exists():
        completed = set(json.loads(checkpoint.read_text(encoding="utf-8")).get("completed", []))

    totals = {"already_present": 0, "updated": 0, "inserted": 0, "orphan": 0, "invalid": 0}
    video_dirs = sorted(path for path in embedding_root.iterdir() if path.is_dir())
    video_dirs = video_dirs[shard_index::shard_count]

    with ThreadPoolExecutor(max_workers=loader_workers) as pool:
        for video_dir in tqdm(video_dirs, desc=f"Ingest {model}", unit="video"):
            video_id = video_dir.name
            if video_id in completed:
                continue

            items = []
            for path in video_dir.glob("keyframe_*.pt"):
                try:
                    frame_idx = int(path.stem.split("_", 1)[1])
                except (IndexError, ValueError):
                    totals["invalid"] += 1
                    continue
                if not (keyframe_root / video_id / f"keyframe_{frame_idx}.webp").exists():
                    totals["orphan"] += 1
                    continue
                items.append((frame_idx, path, _point_id(video_id, frame_idx)))
            items.sort(key=lambda item: item[0])

            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                ids = [item[2] for item in batch]
                records = client.retrieve(
                    collection_name=QDRANT_COLLECTION_NAME,
                    ids=ids,
                    with_payload=False,
                    with_vectors=[model],
                )
                records_by_id = {str(record.id): record for record in records}

                needed = []
                for item in batch:
                    record = records_by_id.get(item[2])
                    vector_names = record.vector if record is not None and isinstance(record.vector, dict) else {}
                    if model in vector_names:
                        totals["already_present"] += 1
                    else:
                        needed.append(item)

                if not needed:
                    continue

                loaded = list(pool.map(lambda item: _load_vector(item[1], expected_dim), needed))
                updates = []
                inserts = []
                for (frame_idx, _path, pid), vector in zip(needed, loaded):
                    if pid in records_by_id:
                        updates.append(PointVectors(id=pid, vector={model: vector}))
                    else:
                        inserts.append(
                            PointStruct(
                                id=pid,
                                vector={model: vector},
                                payload={"video_id": video_id, "keyframe_idx": frame_idx},
                            )
                        )

                if updates:
                    client.update_vectors(
                        collection_name=QDRANT_COLLECTION_NAME,
                        points=updates,
                        wait=wait,
                    )
                    totals["updated"] += len(updates)
                if inserts:
                    client.upsert(
                        collection_name=QDRANT_COLLECTION_NAME,
                        points=inserts,
                        wait=wait,
                    )
                    totals["inserted"] += len(inserts)

            completed.add(video_id)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(
                json.dumps({"completed": sorted(completed), "totals": totals}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3_VL_Embedding")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=Path(DATA_DIR) / ".agent_tmp" / "qwen_ingest_checkpoint.json")
    parser.add_argument(
        "--bulk",
        action="store_true",
        help="Temporarily defer HNSW indexing during ingestion, then restore its previous threshold.",
    )
    parser.add_argument(
        "--async-writes",
        action="store_true",
        help="Queue writes without waiting for each batch to be committed.",
    )
    args = parser.parse_args()
    if args.model not in VISION_EMBEDDING_DIM:
        parser.error(f"Unknown model: {args.model}")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require 0 <= --shard-index < --shard-count")

    client = QdrantClient(url=QDRANT_HOST_URL, prefer_grpc=True, timeout=180)
    previous_threshold = None
    if args.bulk:
        previous_threshold = client.get_collection(QDRANT_COLLECTION_NAME).config.optimizer_config.indexing_threshold
        client.update_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
    try:
        result = ingest(
            args.model,
            args.batch_size,
            args.loader_workers,
            args.checkpoint,
            wait=not args.async_writes,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        print(json.dumps(result, indent=2))
    finally:
        if previous_threshold is not None:
            client.update_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                optimizers_config=OptimizersConfigDiff(indexing_threshold=previous_threshold),
            )


if __name__ == "__main__":
    main()
