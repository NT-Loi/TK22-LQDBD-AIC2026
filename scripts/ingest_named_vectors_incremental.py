"""Resume-safe ingestion of one named vision vector into an existing Qdrant collection.

Unlike ``RetrievalSystem.ingest_vision_embedding``, this utility preserves the other
named vectors already stored on a point. Existing points are updated with
``update_vectors``; missing points are inserted with ``upsert``.

Supports both:
- Directory of per-frame PyTorch tensors: ``<model>/<video_id>/keyframe_<idx>.pt``
- Consolidated NumPy arrays with CSV maps: ``<model>/<video_id>.npy`` + ``maps/<video_id>.csv``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    DenseVectorConfig,
    DenseVectorNameConfig,
    Distance,
    OptimizersConfigDiff,
    PointStruct,
    PointVectors,
)
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATA_DIR,
    EMBEDDING_DIRS,
    KEYFRAME_DIRS,
    MAP_DIRS,
    VIDEO_DIRS,
    QDRANT_COLLECTION_NAME,
    QDRANT_HOST_URL,
    VISION_EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


def _point_id(video_id: str, frame_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{frame_idx}"))


def _load_vector(path: Path, expected_dim: int) -> list[float]:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    vector = tensor.detach().float().reshape(-1)
    if vector.numel() != expected_dim:
        raise ValueError(f"{path}: expected {expected_dim} values, got {vector.numel()}")
    return vector.numpy().tolist()


def _has_keyframe(video_id: str, frame_idx: int) -> bool:
    for kdir in KEYFRAME_DIRS:
        if (Path(kdir) / video_id / f"keyframe_{frame_idx}.webp").exists():
            return True
    return False


def _find_map_csv(video_id: str, map_dirs: list[Path]) -> Path | None:
    cand_stems = [video_id]
    if "_" in video_id:
        cand_stems.append(video_id.replace("_", "-"))
    if "-" in video_id:
        cand_stems.append(video_id.replace("-", "_"))

    for mdir in map_dirs:
        mdir_path = Path(mdir)
        if not mdir_path.is_dir():
            continue
        for stem in cand_stems:
            p = mdir_path / f"{stem}.csv"
            if p.is_file():
                return p
    return None


def ingest(
    model: str,
    batch_size: int,
    loader_workers: int,
    checkpoint: Path,
    wait: bool = True,
    shard_index: int = 0,
    shard_count: int = 1,
    extra_embedding_dirs: list[Path] | None = None,
    require_keyframe: bool = False,
) -> dict:
    expected_dim = VISION_EMBEDDING_DIM[model]
    client = QdrantClient(url=QDRANT_HOST_URL, prefer_grpc=True, timeout=180)

    # 1. Ensure the named vector exists on the Qdrant collection schema
    try:
        coll = client.get_collection(QDRANT_COLLECTION_NAME)
        existing_vectors = coll.config.params.vectors or {}
        if model not in existing_vectors:
            print(f"Creating named vector '{model}' (dim={expected_dim}) in collection '{QDRANT_COLLECTION_NAME}'...")
            client.create_vector_name(
                collection_name=QDRANT_COLLECTION_NAME,
                vector_name=model,
                vector_name_config=DenseVectorNameConfig(
                    dense=DenseVectorConfig(size=expected_dim, distance=Distance.COSINE)
                ),
                wait=True,
            )
    except Exception as e:
        print(f"Warning: unable to verify/create vector '{model}' in collection: {e}")

    completed = set()
    if checkpoint is not None and checkpoint.exists():
        try:
            completed = set(json.loads(checkpoint.read_text(encoding="utf-8")).get("completed", []))
        except Exception:
            pass

    totals = {"already_present": 0, "updated": 0, "inserted": 0, "orphan": 0, "invalid": 0}

    # 2. Discover video sources across all configured EMBEDDING_DIRS
    search_dirs = list(EMBEDDING_DIRS)
    if extra_embedding_dirs:
        for ed in extra_embedding_dirs:
            if ed not in search_dirs:
                search_dirs.append(ed)

    video_sources_map: dict[str, Path] = {}
    for emb_dir in search_dirs:
        m_dir = Path(emb_dir) / model
        if not m_dir.is_dir():
            continue
        for path in m_dir.iterdir():
            if path.is_dir() and path.name not in video_sources_map:
                video_sources_map[path.name] = path
            elif path.is_file() and path.suffix == ".npy" and path.stem not in video_sources_map:
                video_sources_map[path.stem] = path

    video_sources = sorted(video_sources_map.items(), key=lambda item: item[0])
    video_sources = video_sources[shard_index::shard_count]

    to_process = [(vid, path) for vid, path in video_sources if vid not in completed]
    num_cached = len(video_sources) - len(to_process)

    if num_cached > 0 and to_process:
        print(f"⏩ Resuming from checkpoint: {num_cached}/{len(video_sources)} videos already cached. Processing remaining {len(to_process)} videos...")

    with ThreadPoolExecutor(max_workers=loader_workers) as pool:
        for video_id, video_path in tqdm(
            to_process,
            desc=f"Ingest {model}",
            unit="video",
            total=len(video_sources),
            initial=num_cached,
        ):
            items = []  # tuple of (frame_idx, vec_or_path, pid, pts_time)

            if video_path.is_dir():
                for path in video_path.glob("keyframe_*.pt"):
                    try:
                        frame_idx = int(path.stem.split("_", 1)[1])
                    except (IndexError, ValueError):
                        totals["invalid"] += 1
                        continue
                    if require_keyframe and not _has_keyframe(video_id, frame_idx):
                        totals["orphan"] += 1
                        continue
                    items.append((frame_idx, path, _point_id(video_id, frame_idx), None))
                items.sort(key=lambda item: item[0])

            elif video_path.is_file() and video_path.suffix == ".npy":
                map_csv = _find_map_csv(video_id, MAP_DIRS)
                if not map_csv:
                    totals["invalid"] += 1
                    continue
                try:
                    df = pd.read_csv(map_csv)
                    if "frame_idx" not in df.columns:
                        totals["invalid"] += 1
                        continue
                    arr = np.load(video_path)
                    if len(arr) != len(df):
                        totals["invalid"] += 1
                        continue

                    for i in range(len(df)):
                        frame_idx = int(df.iloc[i]["frame_idx"])
                        pts_time = float(df.iloc[i]["pts_time"]) if "pts_time" in df.columns and pd.notna(df.iloc[i]["pts_time"]) else None
                        if require_keyframe and not _has_keyframe(video_id, frame_idx):
                            totals["orphan"] += 1
                            continue
                        vec = arr[i].astype(float).tolist()
                        if len(vec) != expected_dim:
                            totals["invalid"] += 1
                            continue
                        items.append((frame_idx, vec, _point_id(video_id, frame_idx), pts_time))
                except Exception:
                    totals["invalid"] += 1
                    continue

            if not items:
                completed.add(video_id)
                continue

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

                if video_path.is_dir():
                    loaded_vectors = list(pool.map(lambda it: _load_vector(it[1], expected_dim), needed))
                else:
                    loaded_vectors = [it[1] for it in needed]

                updates = []
                inserts = []
                for (frame_idx, _vec_or_path, pid, pts_time), vector in zip(needed, loaded_vectors):
                    if pid in records_by_id:
                        updates.append(PointVectors(id=pid, vector={model: vector}))
                    else:
                        payload = {"video_id": video_id, "keyframe_idx": frame_idx}
                        if pts_time is not None:
                            payload["pts_time"] = pts_time
                        inserts.append(
                            PointStruct(
                                id=pid,
                                vector={model: vector},
                                payload=payload,
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
            if checkpoint is not None:
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Custom checkpoint file path. Defaults to data/.agent_tmp/<model>_ingest_checkpoint.json.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Ignore checkpoint and check all points against Qdrant directly.",
    )
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
    parser.add_argument(
        "--extra-dir",
        type=Path,
        action="append",
        default=None,
        help="Additional root or embedding directory to search (can be specified multiple times).",
    )
    parser.add_argument(
        "--require-keyframe",
        action="store_true",
        default=False,
        help="Only ingest frames whose keyframe .webp image already exists on disk (default: False, ingest all frames).",
    )
    parser.add_argument(
        "--skip-keyframe-check",
        action="store_true",
        default=True,
        help="Deprecated alias for default behavior (ingest all frames).",
    )
    args = parser.parse_args()
    if args.model not in VISION_EMBEDDING_DIM:
        parser.error(f"Unknown model: {args.model}")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require 0 <= --shard-index < --shard-count")

    checkpoint_path = args.checkpoint
    if checkpoint_path is None and not args.no_checkpoint:
        checkpoint_path = Path(DATA_DIR) / ".agent_tmp" / f"{args.model.lower()}_ingest_checkpoint.json"

    client = QdrantClient(url=QDRANT_HOST_URL, prefer_grpc=True, timeout=180)
    previous_threshold = None
    if args.bulk:
        previous_threshold = client.get_collection(QDRANT_COLLECTION_NAME).config.optimizer_config.indexing_threshold
        client.update_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
    try:
        extra_dirs = []
        if args.extra_dir:
            for ed in args.extra_dir:
                if (ed / "embedding").is_dir():
                    extra_dirs.append(ed / "embedding")
                else:
                    extra_dirs.append(ed)

        result = ingest(
            args.model,
            args.batch_size,
            args.loader_workers,
            checkpoint=checkpoint_path if not args.no_checkpoint else None,
            wait=not args.async_writes,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            extra_embedding_dirs=extra_dirs if extra_dirs else None,
            require_keyframe=args.require_keyframe,
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
