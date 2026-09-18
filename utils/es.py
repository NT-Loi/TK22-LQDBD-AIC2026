from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_es_index(es_client: Elasticsearch, index_name: str, overwrite: bool = False):
    """
    Create the Elasticsearch index for OCR text with custom analyzers
    supporting exact, fuzzy, and edge-ngram partial matching.
    """
    if es_client.indices.exists(index=index_name):
        if overwrite:
            logger.info(f"ES index '{index_name}' exists. Deleting...")
            es_client.indices.delete(index=index_name, request_timeout=300)
        else:
            logger.info(f"ES index '{index_name}' already exists. Skipping creation.")
            return

    index_body = {
        "settings": {
            "analysis": {
                "tokenizer": {
                    "ocr_edge_ngram_tokenizer": {
                        "type": "edge_ngram",
                        "min_gram": 2,
                        "max_gram": 15,
                        "token_chars": ["letter", "digit"]
                    }
                },
                "analyzer": {
                    "ocr_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"]
                    },
                    "ocr_ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "ocr_edge_ngram_tokenizer",
                        "filter": ["lowercase", "asciifolding"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "video_id": {"type": "keyword"},
                "keyframe_idx": {"type": "integer"},
                "ocr_source": {"type": "keyword"},
                "ocr_text": {
                    "type": "text",
                    "analyzer": "ocr_analyzer",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "ocr_ngram_analyzer"
                        }
                    }
                }
            }
        }
    }

    es_client.indices.create(index=index_name, body=index_body, request_timeout=300)
    logger.info(f"ES index '{index_name}' created successfully with N-Gram support.")


def ingest_ocr_to_es(es_client: Elasticsearch, index_name: str, ocr_dir: str, ocr_sources: list = None):
    """
    Ingest OCR JSON files into Elasticsearch.
    
    Directory structure: ocr_dir/{ocr_source}/{video_id}/{keyframe_id}.json
    Supports both V6 (rec_texts) and VL1.6 (parsing_res_list) JSON formats.
    """
    ocr_path = Path(ocr_dir)
    if not ocr_sources:
        ocr_sources = [d.name for d in ocr_path.iterdir() if d.is_dir()]

    total_indexed = 0

    for source in ocr_sources:
        source_dir = ocr_path / source
        if not source_dir.exists():
            logger.warning(f"OCR source directory '{source_dir}' not found. Skipping.")
            continue

        actions = []
        video_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
        for video_dir in tqdm(video_dirs, desc=f"📝 OCR [{source}]", unit="video"):
            video_id = video_dir.name

            for file_name in os.listdir(video_dir):
                if not file_name.endswith(".json"):
                    continue

                # Parse keyframe_idx from filename: keyframe_{keyframe_id}.json or {keyframe_id}.json
                clean_name = file_name.replace("keyframe_", "").replace(".json", "")
                try:
                    keyframe_idx = int(clean_name)
                except ValueError:
                    logger.warning(f"Skipping file with unexpected name: {file_name}")
                    continue

                json_path = video_dir / file_name
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to read {json_path}: {e}")
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

                # Join all recognized text lines into a single string
                ocr_text = " ".join(rec_texts)

                # Deterministic doc ID: source_video_keyframe
                doc_id = f"{source}_{video_id}_{keyframe_idx}"

                actions.append({
                    "_index": index_name,
                    "_id": doc_id,
                    "_source": {
                        "video_id": video_id,
                        "keyframe_idx": keyframe_idx,
                        "ocr_source": source,
                        "ocr_text": ocr_text
                    }
                })

                # Bulk index every 2000 docs
                if len(actions) >= 2000:
                    success, _ = bulk(es_client, actions, raise_on_error=False)
                    total_indexed += success
                    actions = []

        # Flush remaining
        if actions:
            success, _ = bulk(es_client, actions, raise_on_error=False)
            total_indexed += success

        logger.info(f"Completed OCR ingestion for source '{source}'. Total indexed so far: {total_indexed}")

    logger.info(f"OCR ingestion complete. Total documents indexed: {total_indexed}")
    return total_indexed


def fuzzy_search_ocr(es_client: Elasticsearch, index_name: str, query: str,
                     ocr_sources: list = None, fuzziness: str = "AUTO", size: int = 5000):
    """
    Perform fuzzy text search on OCR data.
    
    If multiple sources (e.g. V6 and VL1.6) match the same (video_id, keyframe_idx),
    the score for that keyframe will be max(v6_score, vl1.6_score).
    """
    # Build match clause with optimized prefix_length and max_expansions
    match_params = {
        "query": query,
    }
    if fuzziness and str(fuzziness).upper() != "0":
        match_params["fuzziness"] = fuzziness
        match_params["prefix_length"] = 2
        match_params["max_expansions"] = 10

    must_clause = {
        "match": {
            "ocr_text": match_params
        }
    }

    es_query = {
        "bool": {
            "must": [must_clause]
        }
    }

    # Optionally filter by OCR source
    if ocr_sources:
        es_query["bool"]["filter"] = {
            "terms": {"ocr_source": ocr_sources}
        }

    response = es_client.search(
        index=index_name,
        query=es_query,
        size=size,
        _source=["video_id", "keyframe_idx", "ocr_source", "ocr_text"]
    )

    frame_scores = {}
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        key = (src["video_id"], src["keyframe_idx"])
        score = float(hit["_score"])
        
        # Max score aggregation between V6 and VL1.6 (or any matching sources)
        if key not in frame_scores or score > frame_scores[key]["es_score"]:
            frame_scores[key] = {
                "video_id": src["video_id"],
                "keyframe_idx": src["keyframe_idx"],
                "ocr_source": src.get("ocr_source"),
                "ocr_text": src.get("ocr_text", ""),
                "es_score": score
            }

    results = list(frame_scores.values())
    results.sort(key=lambda x: x["es_score"], reverse=True)
    return results


# ─── Transcript (Whisper) ES functions ──────────────────────────────────────

def setup_transcript_index(es_client: Elasticsearch, index_name: str, overwrite: bool = False):
    """
    Create the Elasticsearch index for transcript segments with custom analyzers
    supporting exact, fuzzy, and edge-ngram partial matching.
    """
    if es_client.indices.exists(index=index_name):
        if overwrite:
            logger.info(f"ES index '{index_name}' exists. Deleting...")
            es_client.indices.delete(index=index_name, request_timeout=300)
        else:
            logger.info(f"ES index '{index_name}' already exists. Skipping creation.")
            return

    index_body = {
        "settings": {
            "analysis": {
                "tokenizer": {
                    "transcript_edge_ngram_tokenizer": {
                        "type": "edge_ngram",
                        "min_gram": 2,
                        "max_gram": 15,
                        "token_chars": ["letter", "digit"]
                    }
                },
                "analyzer": {
                    "transcript_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"]
                    },
                    "transcript_ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "transcript_edge_ngram_tokenizer",
                        "filter": ["lowercase", "asciifolding"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "video_id": {"type": "keyword"},
                "segment_id": {"type": "integer"},
                "start": {"type": "float"},
                "end": {"type": "float"},
                "text": {
                    "type": "text",
                    "analyzer": "transcript_analyzer",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "transcript_ngram_analyzer"
                        }
                    }
                }
            }
        }
    }

    es_client.indices.create(index=index_name, body=index_body, request_timeout=300)
    logger.info(f"ES transcript index '{index_name}' created successfully with N-Gram support.")


def ingest_transcript_to_es(es_client: Elasticsearch, index_name: str, transcript_dir: str):
    """
    Ingest Whisper transcript JSON files into Elasticsearch.

    Directory structure: transcript_dir/{video_id}.json
    Each JSON has a 'segments' list with {segment_id, start, end, text, ...}.
    """
    transcript_path = Path(transcript_dir)
    if not transcript_path.exists():
        logger.warning(f"Transcript directory '{transcript_path}' not found.")
        return 0

    total_indexed = 0
    actions = []

    json_files = [f for f in os.listdir(transcript_path) if f.endswith(".json")]
    for file_name in tqdm(json_files, desc="🎙️ Transcripts", unit="video"):
        video_id = file_name.replace(".json", "")
        json_path = transcript_path / file_name

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {json_path}: {e}")
            continue

        segments = data.get("segments", [])
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            seg_id = seg.get("segment_id", 0)
            doc_id = f"{video_id}_{seg_id}"

            actions.append({
                "_index": index_name,
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

    logger.info(f"Transcript ingestion complete. Total segments indexed: {total_indexed}")
    return total_indexed


def fuzzy_search_transcript(es_client: Elasticsearch, index_name: str, query: str,
                            fuzziness: str = "AUTO", size: int = 500):
    """
    Perform fuzzy text search on transcript segments.

    Returns a list of dicts with {video_id, start, end, es_score}
    representing time ranges where the queried text was spoken.
    """
    match_params = {
        "query": query,
    }
    if fuzziness and str(fuzziness).upper() != "0":
        match_params["fuzziness"] = fuzziness
        match_params["prefix_length"] = 2
        match_params["max_expansions"] = 10

    es_query = {
        "bool": {
            "must": [{
                "match": {
                    "text": match_params
                }
            }]
        }
    }

    response = es_client.search(
        index=index_name,
        query=es_query,
        size=size,
        _source=["video_id", "start", "end", "text"]
    )

    results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "video_id": src["video_id"],
            "start": src["start"],
            "end": src["end"],
            "text": src.get("text", ""),
            "es_score": hit["_score"]
        })

    return results

