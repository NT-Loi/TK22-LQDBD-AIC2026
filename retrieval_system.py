from data_processor.text_encoder import TextEncoder
import logging
import sys

# Configure logging to output to both console and a file called 'app.log'
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

import os
import torch
import uuid
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, Range, SetPayloadOperation, SetPayload
from elasticsearch import Elasticsearch

from data_processor.text_encoder import *

from config import DATA_DIR, QDRANT_HOST_URL, ES_HOST_URL, ES_INDEX_NAME, ES_TRANSCRIPT_INDEX_NAME, QDRANT_COLLECTION_NAME, VECTOR_SIZES, EMBEDDING_WEIGHTS, MAX_FRAME_GAP, OCR_SOURCES
from utils import (setup_qdrant_collection, setup_es_index, ingest_ocr_to_es, fuzzy_search_ocr,
                   setup_transcript_index, ingest_transcript_to_es, fuzzy_search_transcript)
from utils.video_metadata import load_video_metadata
import json
import bisect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RetrievalSystem:
    def __init__(self, data_dir: str=DATA_DIR, device=None, re_ingest: bool=False):
        logger.info("Initializing Video Retrieval System...")

        # auto use gpu
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.data_dir = Path(data_dir)
        self.embedding_dir = self.data_dir / "embedding"

        self.qdrant_client = QdrantClient(url=QDRANT_HOST_URL, timeout=60.0)
        self.qdrant_collection_name = QDRANT_COLLECTION_NAME
        if not self.qdrant_client.collection_exists(self.qdrant_collection_name):
            logger.warning(f"Qdrant collection '{self.qdrant_collection_name}' does not exist. Please run ingestion.")

        self.es_client = Elasticsearch(ES_HOST_URL, request_timeout=30)
        self.es_index_name = ES_INDEX_NAME
        self.es_transcript_index_name = ES_TRANSCRIPT_INDEX_NAME

        # Load video metadata (FPS) for frame-to-time conversion
        self.video_metadata = load_video_metadata()

        if re_ingest:
            self.ingest()

        self.text_encoders = {}
        logger.info("Initializing text encoders...")
        for model_name in VECTOR_SIZES.keys():
            if model_name == "CLIP_H14":
                self.text_encoders[model_name] = CLIPTextEncoder(device=self.device)

            if model_name == "SigLIP":
                self.text_encoders[model_name] = SigLIPTextEncoder(device=self.device)

            if model_name == "SigLIP2":
                self.text_encoders[model_name] = SigLIP2TextEncoder(device=self.device)
        # Load shot boundaries
        self.shots_data = {}
        self._load_shots()

    def _load_shots(self):
        shot_dir = self.data_dir / "shot"
        if not shot_dir.exists():
            logger.warning(f"Shot directory {shot_dir} does not exist.")
            return
            
        logger.info("Loading shot boundaries...")
        for filename in os.listdir(shot_dir):
            if filename.endswith(".json"):
                try:
                    with open(shot_dir / filename, "r") as f:
                        data = json.load(f)
                        for video_key, boundaries in data.items():
                            video_id = video_key.split(".")[0]
                            # Sort just in case
                            self.shots_data[video_id] = sorted(boundaries, key=lambda x: x[0])
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")

    def process_video_data(self):
        pass

    def ingest_embedding(self, embedding_model: str):
        model_embedding_dir = self.embedding_dir / f"{embedding_model}"
        if not model_embedding_dir.exists() or not model_embedding_dir.is_dir():
            model_embedding_dir = self.embedding_dir
        if not model_embedding_dir.exists() or not model_embedding_dir.is_dir():
            logger.warning(f"Embedding directory {model_embedding_dir} not found. Skipping {embedding_model}.")
            return

        points = []
        total_ingested = 0
        # Directory structure: embedding_dir / model_name / video_id / keyframe_<idx>.pt
        video_dirs = [d for d in model_embedding_dir.iterdir() if d.is_dir()]
        for video_dir in tqdm(video_dirs, desc=f"📦 Embeddings [{embedding_model}]", unit="video"):
            video_id = video_dir.name
                
            for file_name in os.listdir(video_dir):
                if not file_name.endswith(".pt") or not file_name.startswith("keyframe_"):
                    continue
                
                # Extract keyframe_idx from "keyframe_<idx>.pt"
                try:
                    keyframe_idx = int(file_name.split("_")[1].split(".")[0])
                except (IndexError, ValueError):
                    logger.warning(f"Skipping file with unexpected name format: {file_name}")
                    continue
                    
                expected_dim = VECTOR_SIZES.get(embedding_model)

                # Load torch tensor and convert to python list
                try:
                    embedding_tensor = torch.load(video_dir / file_name, map_location="cpu", weights_only=True)
                    embedding = embedding_tensor.squeeze().tolist()
                except Exception as e:
                    logger.error(f"Failed to load {file_name} for video {video_id}: {e}")
                    continue

                if len(embedding) != expected_dim:
                    continue
                
                # Deterministic UUID based on video_id and keyframe_idx
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{keyframe_idx}"))
                
                points.append(PointStruct(
                    id=point_id,
                    vector={embedding_model: embedding}, # Named vector
                    payload={
                        "video_id": video_id,
                        "keyframe_idx": keyframe_idx
                    }
                ))
                
                # Batch upsert every 1000 points
                if len(points) >= 1000:
                    self.qdrant_client.upsert(
                        collection_name=self.qdrant_collection_name,
                        points=points
                    )
                    total_ingested += len(points)
                    points = []
        
        # Upsert any remaining points
        if points:
            self.qdrant_client.upsert(
                collection_name=self.qdrant_collection_name,
                points=points
            )
            total_ingested += len(points)
        logger.info(f"Ingested {total_ingested} embeddings for model '{embedding_model}' into Qdrant.")

    def ingest_object_detection(self):
        obj_dir = self.data_dir / "object_detection"
        if not obj_dir.exists():
            logger.warning(f"Object detection directory {obj_dir} not found. Skipping.")
            return

        logger.info("Ingesting object detection metadata into Qdrant in batches...")
        count = 0
        operations = []

        def _flush_operations():
            nonlocal operations, count
            if not operations:
                return
            try:
                self.qdrant_client.batch_update_points(
                    collection_name=self.qdrant_collection_name,
                    update_operations=operations
                )
                count += len(operations)
            except Exception as e:
                logger.error(f"Failed batch update for object detection payloads: {e}")
            operations = []

        # Collect all JSON files first for tqdm
        all_json_files = []
        for root, _, files in os.walk(obj_dir):
            for file_name in files:
                if file_name.endswith(".json"):
                    all_json_files.append(Path(root) / file_name)

        for json_path in tqdm(all_json_files, desc="🔍 Object Detection", unit="file"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read JSON {json_path}: {e}")
                continue
                
            image_path = data.get("image_path", "")
            if image_path:
                parent_name = Path(image_path).parent.name
                video_id = parent_name if parent_name and parent_name != "." else data.get("video_id")
            else:
                video_id = json_path.parent.name if json_path.parent.name else data.get("video_id")

            keyframe_idx = data.get("keyframe_index")
            if not video_id or keyframe_idx is None:
                continue

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{keyframe_idx}"))

            payload = {
                "objects": data.get("object_names", []),
                "has_objects": data.get("has_objects", False),
                "num_detections": data.get("num_detections", 0),
                "class_counts": data.get("class_counts", {})
            }

            operations.append(
                SetPayloadOperation(
                    set_payload=SetPayload(
                        payload=payload,
                        points=[point_id]
                    )
                )
            )

            if len(operations) >= 500:
                _flush_operations()

        _flush_operations()
        logger.info(f"Completed object detection payload ingestion for {count} points.")
    
    def ingest_ocr(self):
        ocr_dir = self.data_dir / "ocr"
        if not ocr_dir.exists():
            logger.warning(f"OCR directory {ocr_dir} not found. Skipping OCR ingestion.")
            return
        setup_es_index(self.es_client, self.es_index_name, overwrite=True)
        ingest_ocr_to_es(self.es_client, self.es_index_name, str(ocr_dir), OCR_SOURCES)

    def ingest_transcript(self):
        transcript_dir = self.data_dir / "transcript"
        if not transcript_dir.exists():
            logger.warning(f"Transcript directory {transcript_dir} not found. Skipping transcript ingestion.")
            return
        setup_transcript_index(self.es_client, self.es_transcript_index_name, overwrite=True)
        ingest_transcript_to_es(self.es_client, self.es_transcript_index_name, str(transcript_dir))

    def ingest(self):
        setup_qdrant_collection(self.qdrant_client, self.qdrant_collection_name, VECTOR_SIZES, overwrite=True)
        for model_name in VECTOR_SIZES.keys():    
            self.ingest_embedding(model_name)
        self.ingest_object_detection()
        self.ingest_ocr()
        self.ingest_transcript()

    def _build_qdrant_filter(self, objects: list) -> Optional[Filter]:
        if not objects:
            return None
            
        must_conditions = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            label = obj.get("label")
            if not label:
                continue
                
            must_conditions.append(
                FieldCondition(
                    key="objects",
                    match=MatchValue(value=label)
                )
            )
            
            min_instances = obj.get("min_instances", 1)
            max_instances = obj.get("max_instances")
            
            range_kwargs = {"gte": min_instances}
            if max_instances is not None and isinstance(max_instances, int):
                range_kwargs["lte"] = max_instances
                
            must_conditions.append(
                FieldCondition(
                    key=f"class_counts.{label}",
                    range=Range(**range_kwargs)
                )
            )
            
        if not must_conditions:
            return None
            
        return Filter(must=must_conditions)

    def encode_query(self, query: str, model_names: list = None):
        """
        Encode query using specified text encoders.
        If model_names is None, it uses all initialized text encoders.
        Returns a dictionary mapping model_name to encoded text vector.
        """
        if model_names is None:
            model_names = list(self.text_encoders.keys())
            
        encoded_vectors = {}
        for model_name in model_names:
            if model_name in self.text_encoders:
                encoded_vectors[model_name] = self.text_encoders[model_name](query)
            else:
                logger.warning(f"Text encoder for '{model_name}' is not initialized.")
                
        return encoded_vectors

    def _matches_object_filters(self, payload: dict, objects: list) -> bool:
        if not objects or not payload:
            return True
        
        frame_objects = payload.get("objects", [])
        class_counts = payload.get("class_counts", {})
        
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            label = obj.get("label")
            if not label:
                continue
                
            if label not in frame_objects:
                return False
                
            min_instances = obj.get("min_instances", 1)
            max_instances = obj.get("max_instances")
            actual_count = class_counts.get(label, 0)
            
            if actual_count < min_instances:
                return False
            if max_instances is not None and isinstance(max_instances, int) and actual_count > max_instances:
                return False
                
        return True

    def ocr_search(self, query: str, ocr_sources: list = None, fuzziness: str = "AUTO", size: int = 500):
        """
        Standalone OCR text search. Returns results in the same format as semantic_search.
        """
        logger.info(f"Performing OCR search for: '{query}'")
        ocr_hits = fuzzy_search_ocr(self.es_client, self.es_index_name, query,
                                    ocr_sources=ocr_sources, fuzziness=fuzziness, size=size)
        
        results = []
        for hit in ocr_hits:
            video_id = hit["video_id"]
            keyframe_idx = hit["keyframe_idx"]
            
            shot_start = 0
            shot_end = 0
            if video_id in self.shots_data:
                boundaries = self.shots_data[video_id]
                starts = [b[0] for b in boundaries]
                idx = bisect.bisect_right(starts, keyframe_idx) - 1
                if idx >= 0 and keyframe_idx <= boundaries[idx][1]:
                    shot_start = boundaries[idx][0]
                    shot_end = boundaries[idx][1]
            
            results.append({
                "video_id": video_id,
                "keyframe_index": keyframe_idx,
                "score": hit["es_score"],
                "shot_start_frame": shot_start,
                "shot_end_frame": shot_end
            })
        
        return results

    def _get_ocr_filter_set(self, ocr_query: str, ocr_sources: list = None, fuzziness: str = "AUTO") -> set:
        """
        Get set of (video_id, keyframe_idx) from OCR fuzzy search for intersection filtering.
        """
        ocr_hits = fuzzy_search_ocr(self.es_client, self.es_index_name, ocr_query,
                                    ocr_sources=ocr_sources, fuzziness=fuzziness, size=5000)
        return {(h["video_id"], h["keyframe_idx"]) for h in ocr_hits}

    def _get_transcript_filter_ranges(self, audio_query: str, fuzziness: str = "AUTO") -> dict:
        """
        Get transcript time ranges from ES fuzzy search.
        Returns dict: {video_id: [(start_sec, end_sec), ...]} for matched segments.
        """
        hits = fuzzy_search_transcript(self.es_client, self.es_transcript_index_name,
                                       audio_query, fuzziness=fuzziness, size=5000)
        ranges = {}
        for h in hits:
            vid = h["video_id"]
            if vid not in ranges:
                ranges[vid] = []
            ranges[vid].append((h["start"], h["end"]))
        return ranges

    def _frame_in_transcript_ranges(self, video_id: str, keyframe_idx: int, transcript_ranges: dict) -> bool:
        """
        Check if a keyframe falls within any matched transcript time range.
        Converts keyframe_idx to seconds using the video's FPS.
        """
        if video_id not in transcript_ranges:
            return False

        # Get FPS for this video (default 25.0)
        fps = 25.0
        if video_id in self.video_metadata and "fps" in self.video_metadata[video_id]:
            fps = self.video_metadata[video_id]["fps"]

        frame_time_sec = keyframe_idx / fps

        for (start_sec, end_sec) in transcript_ranges[video_id]:
            if start_sec <= frame_time_sec <= end_sec:
                return True
        return False

    def filter_search(self, ocr_query: str = None, audio_query: str = None, objects: list = None,
                      group_by_shot: bool = False, score_threshold: float = 0.0, limit: int = 100):
        """
        Perform search purely using metadata filters (OCR, audio transcript, object detection)
        without requiring a text query.
        """
        logger.info(f"Performing filter search with ocr_query='{ocr_query}', audio_query='{audio_query}', objects={objects}")

        if not ocr_query and not audio_query and not objects:
            return []

        # 1. Gather transcript ranges if audio_query is given
        transcript_ranges = None
        if audio_query:
            transcript_ranges = self._get_transcript_filter_ranges(audio_query)
            if not transcript_ranges and not ocr_query and not objects:
                return []

        # Map: (video_id, keyframe_idx) -> score
        candidates = {}

        if ocr_query:
            # Case A: OCR query provided
            ocr_hits = fuzzy_search_ocr(self.es_client, self.es_index_name, ocr_query, size=2000)
            
            for hit in ocr_hits:
                vid = hit["video_id"]
                kf_idx = hit["keyframe_idx"]
                score = hit["es_score"]
                
                # Check transcript filter if audio_query present
                if transcript_ranges is not None and not self._frame_in_transcript_ranges(vid, kf_idx, transcript_ranges):
                    continue

                candidates[(vid, kf_idx)] = score

            # Check object filter if objects present
            if objects and candidates:
                filtered_candidates = {}
                for (vid, kf_idx), score in candidates.items():
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{vid}_{kf_idx}"))
                    try:
                        retrieved = self.qdrant_client.retrieve(
                            collection_name=self.qdrant_collection_name,
                            ids=[point_id],
                            with_payload=True
                        )
                        if retrieved and self._matches_object_filters(retrieved[0].payload, objects):
                            filtered_candidates[(vid, kf_idx)] = score
                    except Exception as e:
                        logger.warning(f"Error retrieving Qdrant point {point_id}: {e}")
                candidates = filtered_candidates

        elif audio_query:
            # Case B: Audio query provided (no OCR query)
            qdrant_filter = self._build_qdrant_filter(objects) if objects else None
            
            points = []
            if qdrant_filter:
                res, _ = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection_name,
                    scroll_filter=qdrant_filter,
                    limit=5000,
                    with_payload=True
                )
                points = res
            else:
                for vid in list(transcript_ranges.keys()):
                    v_filter = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=vid))])
                    res, _ = self.qdrant_client.scroll(
                        collection_name=self.qdrant_collection_name,
                        scroll_filter=v_filter,
                        limit=1000,
                        with_payload=True
                    )
                    points.extend(res)

            for p in points:
                vid = p.payload.get("video_id")
                kf_idx = p.payload.get("keyframe_idx", p.payload.get("frame_idx", 0))
                if vid and kf_idx is not None:
                    if self._frame_in_transcript_ranges(vid, kf_idx, transcript_ranges):
                        candidates[(vid, kf_idx)] = 1.0

        elif objects:
            # Case C: Only object filter provided (no OCR, no audio)
            qdrant_filter = self._build_qdrant_filter(objects)
            points, _ = self.qdrant_client.scroll(
                collection_name=self.qdrant_collection_name,
                scroll_filter=qdrant_filter,
                limit=limit * 5 if group_by_shot else limit,
                with_payload=True
            )
            for p in points:
                vid = p.payload.get("video_id")
                kf_idx = p.payload.get("keyframe_idx", p.payload.get("frame_idx", 0))
                if vid and kf_idx is not None:
                    candidates[(vid, kf_idx)] = 1.0

        # Build formatted output results
        results = []
        for (vid, kf_idx), score in candidates.items():
            if score_threshold > 0 and score < score_threshold and score < 1.0:
                continue

            shot_start = 0
            shot_end = 0
            if vid in self.shots_data:
                boundaries = self.shots_data[vid]
                starts = [b[0] for b in boundaries]
                idx = bisect.bisect_right(starts, kf_idx) - 1
                if idx >= 0 and kf_idx <= boundaries[idx][1]:
                    shot_start = boundaries[idx][0]
                    shot_end = boundaries[idx][1]

            results.append({
                "video_id": vid,
                "keyframe_index": kf_idx,
                "score": score,
                "shot_start_frame": shot_start,
                "shot_end_frame": shot_end
            })

        if group_by_shot:
            shot_dict = {}
            for item in results:
                key = (item["video_id"], item["shot_start_frame"], item["shot_end_frame"])
                if key not in shot_dict:
                    shot_dict[key] = []
                shot_dict[key].append(item)

            grouped_results = []
            for (vid, start, end), items in shot_dict.items():
                avg_score = sum(i["score"] for i in items) / len(items)
                items.sort(key=lambda x: x["keyframe_index"])
                anchor = items[0]
                grouped_results.append({
                    "type": "shot",
                    "video_id": vid,
                    "score": avg_score,
                    "frames": items,
                    "keyframe_index": anchor["keyframe_index"],
                    "shot_start_frame": start,
                    "shot_end_frame": end
                })

            grouped_results.sort(key=lambda x: x["score"], reverse=True)
            return grouped_results[:limit]

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def semantic_search(self, query: str, model_names: list = None, objects: list = None, top_k: int = 1000, score_threshold: float = 0.0, group_by_shot: bool = False, limit: int = 100, ocr_query: str = None, audio_query: str = None):
        logger.info(f"Performing semantic search for: '{query}' with model(s): {model_names}, ocr_query: '{ocr_query}', audio_query: '{audio_query}'.")
        
        if not query or not query.strip():
            return self.filter_search(
                ocr_query=ocr_query,
                audio_query=audio_query,
                objects=objects,
                group_by_shot=group_by_shot,
                score_threshold=score_threshold,
                limit=limit
            )
        
        # Build OCR filter set if ocr_query is provided
        ocr_filter_set = None
        if ocr_query:
            ocr_filter_set = self._get_ocr_filter_set(ocr_query)

        # Build transcript filter ranges if audio_query is provided
        transcript_ranges = None
        if audio_query:
            transcript_ranges = self._get_transcript_filter_ranges(audio_query)
        
        encoded_vectors = self.encode_query(query, model_names)
        
        normalized_results = {}
        all_payloads = {}
        
        for model_name, vector in encoded_vectors.items():
            if model_name not in EMBEDDING_WEIGHTS:
                logger.warning(f"Skipping '{model_name}' search because it has no weight in EMBEDDING_WEIGHTS.")
                continue
                
            query_vector = vector[0].tolist()
            
            search_result = self.qdrant_client.query_points(
                collection_name=self.qdrant_collection_name,
                query=query_vector,
                using=model_name,
                limit=top_k,
                with_payload=True
            ).points

            if objects:
                search_result = [hit for hit in search_result if self._matches_object_filters(hit.payload, objects)]
            
            if not search_result:
                continue
                
            # Min-Max Normalization
            scores = [hit.score for hit in search_result]
            min_score = min(scores)
            max_score = max(scores)
            
            model_normalized_scores = {}
            for hit in search_result:
                # Save payload for final output
                all_payloads[hit.id] = hit.payload
                
                # Handle edge case where max == min
                if max_score == min_score:
                    norm_score = 1.0 if max_score > 0 else 0.0
                else:
                    norm_score = (hit.score - min_score) / (max_score - min_score)
                    
                model_normalized_scores[hit.id] = norm_score
                
            normalized_results[model_name] = model_normalized_scores
            
        # Calculate overall score
        final_scores = {}
        total_weight = sum(EMBEDDING_WEIGHTS.values())
        for model_name, scores_dict in normalized_results.items():
            weight = EMBEDDING_WEIGHTS.get(model_name)
            for point_id, norm_score in scores_dict.items():
                if point_id not in final_scores:
                    final_scores[point_id] = 0.0
                final_scores[point_id] += float(norm_score * weight / total_weight)
                
        # Apply OCR intersection filter before ranking
        if ocr_filter_set is not None:
            final_scores = {
                pid: score for pid, score in final_scores.items()
                if (all_payloads[pid].get("video_id"), all_payloads[pid].get("keyframe_idx", all_payloads[pid].get("frame_idx"))) in ocr_filter_set
            }
            logger.info(f"After OCR filter: {len(final_scores)} results remain.")

        # Apply transcript intersection filter before ranking
        if transcript_ranges is not None:
            final_scores = {
                pid: score for pid, score in final_scores.items()
                if self._frame_in_transcript_ranges(
                    all_payloads[pid].get("video_id"),
                    all_payloads[pid].get("keyframe_idx", all_payloads[pid].get("frame_idx", 0)),
                    transcript_ranges
                )
            }
            logger.info(f"After transcript filter: {len(final_scores)} results remain.")

        # Rank and return
        ranked_points = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for point_id, total_score in ranked_points:
            if total_score < score_threshold:
                continue
                
            payload = all_payloads[point_id]
            video_id = payload.get("video_id")
            keyframe_idx = payload.get("keyframe_idx", payload.get("frame_idx"))
            
            shot_start = 0
            shot_end = 0
            if video_id in self.shots_data:
                # Binary search to find the shot containing keyframe_idx
                boundaries = self.shots_data[video_id]
                # Extract just the start frames for bisect
                starts = [b[0] for b in boundaries]
                idx = bisect.bisect_right(starts, keyframe_idx) - 1
                if idx >= 0 and keyframe_idx <= boundaries[idx][1]:
                    shot_start = boundaries[idx][0]
                    shot_end = boundaries[idx][1]
            
            results.append({
                "video_id": video_id,
                "keyframe_index": keyframe_idx,
                "score": total_score,
                "shot_start_frame": shot_start,
                "shot_end_frame": shot_end
            })
            
        if group_by_shot:
            shot_dict = {}
            for item in results:
                key = (item["video_id"], item["shot_start_frame"], item["shot_end_frame"])
                if key not in shot_dict:
                    shot_dict[key] = []
                shot_dict[key].append(item)
                
            grouped_results = []
            for (vid, start, end), items in shot_dict.items():
                avg_score = sum(i["score"] for i in items) / len(items)
                items.sort(key=lambda x: x["keyframe_index"])
                anchor = items[0]
                grouped_results.append({
                    "type": "shot",
                    "video_id": vid,
                    "score": avg_score,
                    "frames": items,
                    "keyframe_index": anchor["keyframe_index"],
                    "shot_start_frame": start,
                    "shot_end_frame": end
                })
                
            grouped_results.sort(key=lambda x: x["score"], reverse=True)
            return grouped_results[:limit]
            
        return results[:limit]

    def temporal_search(self, queries: list, model_names: list = None, objects: list = None, group_by_shot: bool = False, score_threshold: float = 0.3, limit: int = 100, ocr_query: str = None, audio_query: str = None):
        logger.info(f"Performing temporal search for {len(queries) if queries else 0} queries with group_by_shot={group_by_shot}")
        
        valid_queries = [q.strip() for q in queries if q and q.strip()] if queries else []
        if not valid_queries:
            return self.filter_search(
                ocr_query=ocr_query,
                audio_query=audio_query,
                objects=objects,
                group_by_shot=group_by_shot,
                score_threshold=score_threshold,
                limit=limit
            )
        
        # 1. Search each query
        query_results = []
        for q in valid_queries:
            # We get more than 1000 to ensure we have enough paths, or keep it 1000
            res = self.semantic_search(q, model_names=model_names, objects=objects, top_k=2000, score_threshold=score_threshold, group_by_shot=group_by_shot, ocr_query=ocr_query, audio_query=audio_query)
            query_results.append(res)
            
        # 2. Group by video
        video_grouped = {}
        for q_idx, res_list in enumerate(query_results):
            for item in res_list:
                vid = item["video_id"]
                if vid not in video_grouped:
                    video_grouped[vid] = [[] for _ in range(len(queries))]
                video_grouped[vid][q_idx].append(item)
                
        # 3. Find valid paths via DFS for each video
        all_sequences = []
        
        for vid, vid_group in video_grouped.items():
            # If any query has no results for this video, skip
            if any(len(q_res) == 0 for q_res in vid_group):
                continue
            
            # Items are already aggregated if group_by_shot=True
            for i in range(len(queries)):
                if group_by_shot:
                    vid_group[i].sort(key=lambda x: x["shot_start_frame"])
                else:
                    vid_group[i].sort(key=lambda x: x["keyframe_index"])
            vid_group_to_search = vid_group
            
            def find_paths(current_q_idx, current_path):
                if current_q_idx == len(queries):
                    all_sequences.append(list(current_path))
                    return
                    
                for candidate in vid_group_to_search[current_q_idx]:
                    if len(current_path) == 0:
                        current_path.append(candidate)
                        find_paths(current_q_idx + 1, current_path)
                        current_path.pop()
                    else:
                        prev = current_path[-1]
                        valid = False
                        
                        if not group_by_shot:
                            gap = candidate["keyframe_index"] - prev["keyframe_index"]
                            if 0 < gap <= MAX_FRAME_GAP:
                                valid = True
                        else:
                            # gap between shots logic
                            if candidate["shot_start_frame"] > prev["shot_end_frame"]:
                                gap = candidate["shot_start_frame"] - prev["shot_end_frame"]
                                if gap <= MAX_FRAME_GAP:
                                    valid = True
                                    
                        if valid:
                            current_path.append(candidate)
                            find_paths(current_q_idx + 1, current_path)
                            current_path.pop()

            find_paths(0, [])
            
        # 4. Format output
        final_results = []
        for seq in all_sequences:
            avg_score = sum(item["score"] for item in seq) / len(seq)
            
            if group_by_shot:
                flat_frames = []
                display_frames = []
                for cand in seq:
                    sorted_items = sorted(cand["frames"], key=lambda x: x["keyframe_index"])
                    flat_frames.extend(sorted_items)
                    # Best frame per shot for card thumbnail
                    best_frame = max(cand["frames"], key=lambda x: x["score"])
                    display_frames.append(best_frame)
                anchor = display_frames[0]
            else:
                flat_frames = seq
                display_frames = seq
                anchor = seq[0]
            
            final_results.append({
                "video_id": anchor["video_id"],
                "sequence_score": avg_score,
                "frames": flat_frames,
                "display_frames": display_frames,
                # Include standard fields for compatibility if needed
                "keyframe_index": anchor["keyframe_index"],
                "shot_start_frame": anchor.get("shot_start_frame"),
                "shot_end_frame": anchor.get("shot_end_frame"),
                "score": avg_score, 
            })
            
        # Sort sequences by avg_score
        final_results = sorted(final_results, key=lambda x: x["sequence_score"], reverse=True)
        return final_results[:limit]

if __name__ == "__main__":
    system = RetrievalSystem(re_ingest=True)
    results = system.semantic_search("A person riding a horse on a beach.")