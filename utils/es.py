from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_ocr_index(es_client: Elasticsearch, index_name: str, overwrite: bool = False):
    """
    Create the Elasticsearch index for OCR text with custom analyzers
    supporting exact, fuzzy, and edge-ngram partial matching.
    """
    if es_client.indices.exists(index=index_name):
        if overwrite:
            logger.info(f"ES index '{index_name}' exists. Deleting...")
            es_client.indices.delete(index=index_name, timeout=300)
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

    es_client.indices.create(index=index_name, body=index_body, timeout=300)
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
        source=["video_id", "keyframe_idx", "ocr_source", "ocr_text"]
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
            es_client.indices.delete(index=index_name, timeout=300)
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

    es_client.indices.create(index=index_name, body=index_body, timeout=300)
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
        source=["video_id", "start", "end", "text"]
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


# ─── Shot Caption ES functions ──────────────────────────────────────────────

def setup_caption_index(es_client: Elasticsearch, index_name: str, overwrite: bool = False):
    """
    Create the Elasticsearch index for shot captions with custom analyzer
    supporting exact and case/accent-insensitive text matching.
    """
    if es_client.indices.exists(index=index_name):
        if overwrite:
            logger.info(f"ES caption index '{index_name}' exists. Deleting...")
            es_client.indices.delete(index=index_name)
        else:
            logger.info(f"ES caption index '{index_name}' already exists. Skipping creation.")
            return

    index_body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "caption_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "video_id": {"type": "keyword"},
                "shot_idx": {"type": "integer"},
                "start_frame": {"type": "integer"},
                "end_frame": {"type": "integer"},
                "representative_frame": {"type": "integer"},
                "start_sec": {"type": "float"},
                "end_sec": {"type": "float"},
                "caption_text": {
                    "type": "text",
                    "analyzer": "caption_analyzer"
                },
                "aspects": {
                    "properties": {
                        "chu_the_hanh_dong": {"type": "text", "analyzer": "caption_analyzer"},
                        "vat_the_dac_diem": {"type": "text", "analyzer": "caption_analyzer"},
                        "boi_canh_khong_gian": {"type": "text", "analyzer": "caption_analyzer"},
                        "chu_logo_man_hinh": {"type": "text", "analyzer": "caption_analyzer"},
                        "goc_nhin_co_canh": {"type": "text", "analyzer": "caption_analyzer"},
                        "dien_bien_thoi_gian": {"type": "text", "analyzer": "caption_analyzer"},
                        "tong_the_canh_quay": {"type": "text", "analyzer": "caption_analyzer"}
                    }
                },
                "ocr_detected": {
                    "type": "text",
                    "analyzer": "caption_analyzer"
                },
                "audio_transcript": {
                    "type": "text",
                    "analyzer": "caption_analyzer"
                }
            }
        }
    }

    es_client.indices.create(index=index_name, body=index_body)
    logger.info(f"ES caption index '{index_name}' created successfully.")


def ingest_caption_to_es(es_client: Elasticsearch, index_name: str, caption_dir: str,
                         caption_embedding_dir: str = None):
    """
    Ingest shot caption JSON files into Elasticsearch.
    Uses representative frames from caption_embedding (.pt) if available,
    otherwise defaults to middle keyframe from sampled_keyframes.
    """
    import torch
    cap_path = Path(caption_dir)
    if not cap_path.exists():
        logger.warning(f"Caption directory '{cap_path}' not found.")
        return 0

    emb_path = Path(caption_embedding_dir) if caption_embedding_dir else None

    total_indexed = 0
    actions = []

    json_files = sorted([f for f in os.listdir(cap_path) if f.endswith(".json")])
    for file_name in tqdm(json_files, desc="📝 Captions to ES", unit="video"):
        video_id = file_name.replace(".json", "")
        json_file = cap_path / file_name

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                cap_json = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {json_file}: {e}")
            continue

        # Optionally load representative frames from .pt if present
        rep_frames_map = {}
        if emb_path and (emb_path / f"{video_id}.pt").exists():
            try:
                emb_data = torch.load(emb_path / f"{video_id}.pt", map_location="cpu", weights_only=True)
                s_indices = emb_data.get("shot_indices", [])
                r_frames = emb_data.get("representative_frames", [])
                for s_i, r_f in zip(s_indices, r_frames):
                    rep_frames_map[s_i] = r_f
            except Exception:
                pass

        shots = cap_json.get("shots", [])
        for s in shots:
            shot_idx = s.get("shot_idx", 0)
            start_frame = s.get("start_frame", 0)
            end_frame = s.get("end_frame", 0)

            # Representative frame
            if shot_idx in rep_frames_map:
                rep_frame = rep_frames_map[shot_idx]
            else:
                sampled = s.get("sampled_keyframes", [])
                if sampled:
                    rep_frame = sampled[len(sampled) // 2]
                else:
                    rep_frame = (start_frame + end_frame) // 2

            ocr_text = " ".join(s.get("ocr_detected", [])) if isinstance(s.get("ocr_detected"), list) else str(s.get("ocr_detected", ""))
            doc_id = f"{video_id}_{shot_idx}"

            actions.append({
                "_index": index_name,
                "_id": doc_id,
                "_source": {
                    "video_id": video_id,
                    "shot_idx": shot_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "representative_frame": rep_frame,
                    "start_sec": s.get("start_sec", 0.0),
                    "end_sec": s.get("end_sec", 0.0),
                    "caption_text": s.get("caption_text", ""),
                    "aspects": s.get("aspects", {}),
                    "ocr_detected": ocr_text,
                    "audio_transcript": s.get("audio_transcript", "")
                }
            })

            if len(actions) >= 2000:
                success, _ = bulk(es_client, actions, raise_on_error=False)
                total_indexed += success
                actions = []

    if actions:
        success, _ = bulk(es_client, actions, raise_on_error=False)
        total_indexed += success

    logger.info(f"Caption ingestion complete. Total shots indexed: {total_indexed}")
    return total_indexed


def search_caption_bm25(es_client: Elasticsearch, index_name: str, query: str, size: int = 1000):
    """
    Perform BM25 text search on shot captions across all aspect fields and caption text
    with equal weights (as configured).
    """
    if not query or not query.strip():
        return []

    if not es_client.indices.exists(index=index_name):
        logger.warning(f"Caption index '{index_name}' does not exist in Elasticsearch.")
        return []

    search_fields = [
        "caption_text",
        "aspects.tong_the_canh_quay",
        "aspects.chu_the_hanh_dong",
        "aspects.vat_the_dac_diem",
        "aspects.boi_canh_khong_gian",
        "aspects.chu_logo_man_hinh",
        "aspects.goc_nhin_co_canh",
        "aspects.dien_bien_thoi_gian",
        "ocr_detected",
        "audio_transcript"
    ]

    es_query = {
        "multi_match": {
            "query": query,
            "fields": search_fields,
            "type": "most_fields"
        }
    }

    try:
        response = es_client.search(
            index=index_name,
            query=es_query,
            size=size
        )
    except Exception as e:
        logger.error(f"ES caption BM25 search failed: {e}")
        return []

    results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "video_id": src["video_id"],
            "shot_idx": src["shot_idx"],
            "start_frame": src.get("start_frame", 0),
            "end_frame": src.get("end_frame", 0),
            "representative_frame": src.get("representative_frame", 0),
            "start_sec": src.get("start_sec", 0.0),
            "end_sec": src.get("end_sec", 0.0),
            "caption_text": src.get("caption_text", ""),
            "aspects": src.get("aspects", {}),
            "ocr_detected": src.get("ocr_detected", ""),
            "audio_transcript": src.get("audio_transcript", ""),
            "bm25_score": float(hit["_score"])
        })

    return results


