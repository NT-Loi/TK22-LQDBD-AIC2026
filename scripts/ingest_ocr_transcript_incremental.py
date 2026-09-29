#!/usr/bin/env python3
"""
Incremental ingestion script for OCR and Transcripts into Elasticsearch.
Only indexes new videos that are not yet present in the Elasticsearch index.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import argparse
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from config import (
    ES_HOST_URL,
    ES_OCR_INDEX_NAME,
    ES_TRANSCRIPT_INDEX_NAME,
    OCR_SOURCES,
    DATA_DIR
)
from utils.es import (
    setup_ocr_index,
    setup_transcript_index
)


def get_indexed_videos(es_client: Elasticsearch, index_name: str) -> set:
    """Retrieve the set of all unique video_ids currently in the ES index."""
    if not es_client.indices.exists(index=index_name):
        return set()
    try:
        res = es_client.search(
            index=index_name,
            body={
                "size": 0,
                "aggs": {
                    "vids": {
                        "terms": {"field": "video_id", "size": 20000}
                    }
                }
            },
            timeout="30s"
        )
        return {b["key"] for b in res["aggregations"]["vids"]["buckets"]}
    except Exception as e:
        print(f"Warning: Failed to fetch indexed videos from {index_name}: {e}")
        return set()


def ingest_transcripts_incremental(es_client: Elasticsearch, transcript_dir: Path, force_all: bool = False):
    """Ingest new transcript JSON files into Elasticsearch."""
    setup_transcript_index(es_client, ES_TRANSCRIPT_INDEX_NAME, overwrite=False)
    
    indexed_vids = set() if force_all else get_indexed_videos(es_client, ES_TRANSCRIPT_INDEX_NAME)
    all_files = [f for f in os.listdir(transcript_dir) if f.endswith(".json")]
    
    files_to_index = [
        f for f in all_files
        if force_all or f.replace(".json", "") not in indexed_vids
    ]
    
    print(f"\n🎙️ [Transcript] Found {len(all_files)} total videos on disk.")
    print(f"🎙️ [Transcript] Already in ES: {len(indexed_vids)} videos.")
    print(f"🎙️ [Transcript] New videos to ingest: {len(files_to_index)} videos.")
    
    if not files_to_index:
        print("🎙️ [Transcript] Everything up-to-date! Skipping.")
        return 0

    total_indexed = 0
    actions = []
    
    t0 = time.time()
    for file_name in tqdm(files_to_index, desc="🎙️ Indexing Transcripts", unit="video"):
        video_id = file_name.replace(".json", "")
        json_path = transcript_dir / file_name

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            continue

        segments = data.get("segments", [])
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            seg_id = seg.get("segment_id", 0)
            doc_id = f"{video_id}_{seg_id}"

            actions.append({
                "_index": ES_TRANSCRIPT_INDEX_NAME,
                "_id": doc_id,
                "_source": {
                    "video_id": video_id,
                    "segment_id": seg_id,
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": text
                }
            })

            if len(actions) >= 2000:
                success, _ = bulk(es_client, actions, raise_on_error=False)
                total_indexed += success
                actions = []

    if actions:
        success, _ = bulk(es_client, actions, raise_on_error=False)
        total_indexed += success

    elapsed = time.time() - t0
    print(f"🎙️ [Transcript] Ingestion finished! Added {total_indexed} segments from {len(files_to_index)} videos in {elapsed:.2f}s.")
    return total_indexed


def ingest_ocr_incremental(es_client: Elasticsearch, ocr_dir: Path, ocr_sources: list = None, force_all: bool = False):
    """Ingest new OCR JSON files into Elasticsearch."""
    setup_ocr_index(es_client, ES_OCR_INDEX_NAME, overwrite=False)
    
    if not ocr_sources:
        ocr_sources = OCR_SOURCES

    indexed_vids = set() if force_all else get_indexed_videos(es_client, ES_OCR_INDEX_NAME)
    print(f"\n📝 [OCR] Already indexed in ES: {len(indexed_vids)} unique videos.")

    total_indexed = 0
    t0 = time.time()

    for source in ocr_sources:
        source_dir = ocr_dir / source
        if not source_dir.exists():
            continue

        video_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
        dirs_to_index = [
            d for d in video_dirs
            if force_all or d.name not in indexed_vids
        ]

        print(f"📝 [OCR - {source}] Total on disk: {len(video_dirs)} videos. New to index: {len(dirs_to_index)} videos.")
        if not dirs_to_index:
            continue

        actions = []
        for video_dir in tqdm(dirs_to_index, desc=f"📝 Indexing OCR [{source}]", unit="video"):
            video_id = video_dir.name

            for file_name in os.listdir(video_dir):
                if not file_name.endswith(".json"):
                    continue

                clean_name = file_name.replace("keyframe_", "").replace(".json", "")
                try:
                    keyframe_idx = int(clean_name)
                except ValueError:
                    continue

                json_path = video_dir / file_name
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                rec_texts = []
                # V6 format: rec_texts list
                if "rec_texts" in data and isinstance(data["rec_texts"], list):
                    rec_texts.extend([t.strip() for t in data["rec_texts"] if isinstance(t, str) and t.strip()])

                # VL1.6 format: parsing_res_list
                if "parsing_res_list" in data and isinstance(data["parsing_res_list"], list):
                    for block in data["parsing_res_list"]:
                        if isinstance(block, dict) and block.get("block_content"):
                            txt = block["block_content"].strip()
                            if txt:
                                rec_texts.append(txt)

                if not rec_texts:
                    continue

                ocr_text = " ".join(rec_texts)
                doc_id = f"{source}_{video_id}_{keyframe_idx}"

                actions.append({
                    "_index": ES_OCR_INDEX_NAME,
                    "_id": doc_id,
                    "_source": {
                        "video_id": video_id,
                        "keyframe_idx": keyframe_idx,
                        "ocr_source": source,
                        "ocr_text": ocr_text
                    }
                })

                if len(actions) >= 2000:
                    success, _ = bulk(es_client, actions, raise_on_error=False)
                    total_indexed += success
                    actions = []

        if actions:
            success, _ = bulk(es_client, actions, raise_on_error=False)
            total_indexed += success

    elapsed = time.time() - t0
    print(f"📝 [OCR] Ingestion finished! Added {total_indexed} keyframe records in {elapsed:.2f}s.")
    return total_indexed


def main():
    parser = argparse.ArgumentParser(description="Incremental ingestion of OCR and Transcripts into Elasticsearch")
    parser.add_argument("--force-all", action="store_true", help="Re-index all videos regardless of existing documents")
    parser.add_argument("--ocr-only", action="store_true", help="Only ingest OCR data")
    parser.add_argument("--transcript-only", action="store_true", help="Only ingest Transcript data")
    args = parser.parse_args()

    es_client = Elasticsearch(ES_HOST_URL, timeout=60)
    if not es_client.ping():
        print(f"❌ Error: Cannot connect to Elasticsearch at {ES_HOST_URL}")
        return

    data_dir = Path(DATA_DIR)
    transcript_dir = data_dir / "transcript"
    ocr_dir = data_dir / "ocr"

    print("==================================================")
    print("🚀 Incremental OCR & Transcript Ingestion Starting")
    print(f"Target Elasticsearch: {ES_HOST_URL}")
    print("==================================================")

    if not args.ocr_only:
        ingest_transcripts_incremental(es_client, transcript_dir, force_all=args.force_all)

    if not args.transcript_only:
        ingest_ocr_incremental(es_client, ocr_dir, force_all=args.force_all)

    # Refresh indices to make docs searchable immediately
    print("\n🔄 Refreshing Elasticsearch indices...")
    es_client.indices.refresh(index=ES_TRANSCRIPT_INDEX_NAME)
    es_client.indices.refresh(index=ES_OCR_INDEX_NAME)

    # Final summary
    ocr_count = es_client.count(index=ES_OCR_INDEX_NAME)["count"]
    tr_count = es_client.count(index=ES_TRANSCRIPT_INDEX_NAME)["count"]
    print("==================================================")
    print("✅ Ingestion Complete!")
    print(f"📊 Total documents in '{ES_OCR_INDEX_NAME}': {ocr_count:,}")
    print(f"📊 Total documents in '{ES_TRANSCRIPT_INDEX_NAME}': {tr_count:,}")
    print("==================================================")


if __name__ == "__main__":
    main()
