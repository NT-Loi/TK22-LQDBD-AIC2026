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
import time
import torch
import uuid
from pathlib import Path
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, Range, SetPayloadOperation, SetPayload
# from elasticsearch import Elasticsearch

from data_processor.text_encoder import *

from config import DATA_DIR, QDRANT_HOST_URL, ES_HOST_URL, QDRANT_COLLECTION_NAME, VECTOR_SIZES, EMBEDDING_WEIGHTS, MAX_FRAME_GAP
from utils import setup_qdrant_collection
import json
import bisect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INGEST_BATCH_SIZE = 256
INGEST_PROGRESS_INTERVAL = 10000
UPSERT_TIMEOUT_SECONDS = 180
UPSERT_MAX_RETRIES = 5

class RetrievalSystem:
    def __init__(self, data_dir: str=DATA_DIR, device=None, re_ingest: bool=False, auto_ingest: bool=False):
        logger.info("Initializing Video Retrieval System...")

        # auto use gpu
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.data_dir = Path(data_dir)
        self.embedding_dir = self.data_dir / "embedding"

        self.qdrant_client = QdrantClient(url=QDRANT_HOST_URL, timeout=UPSERT_TIMEOUT_SECONDS)
        self.qdrant_collection_name = QDRANT_COLLECTION_NAME

        # self.es_client = Elasticsearch(ES_HOST_URL, request_timeout=30)

        needs_ingest = self._needs_ingest()
        if re_ingest:
            self.ingest()
        elif auto_ingest and needs_ingest:
            logger.info("Qdrant collection is missing or empty. Running ingestion...")
            self.ingest()
        elif needs_ingest:
            logger.warning(
                "Qdrant collection is missing or empty. Start the app with --ingest "
                "to load embeddings before searching."
            )

        self.text_encoders = {}
        logger.info("Initializing text encoders...")
        for model_name in VECTOR_SIZES.keys():
            if model_name == "CLIP_H14":
                self.text_encoders[model_name] = CLIPTextEncoder(device=self.device)

            if model_name == "SigLIP":
                self.text_encoders[model_name] = SigLIPTextEncoder(device=self.device)

        # Load shot boundaries
        self.shots_data = {}
        self._load_shots()

    def _needs_ingest(self) -> bool:
        if not self.qdrant_client.collection_exists(self.qdrant_collection_name):
            return True

        try:
            count_result = self.qdrant_client.count(
                collection_name=self.qdrant_collection_name,
                exact=True,
            )
            return count_result.count == 0
        except Exception as e:
            logger.warning(f"Could not check Qdrant collection count: {e}")
            return False

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

    def ingest(self):
        setup_qdrant_collection(self.qdrant_client, self.qdrant_collection_name, VECTOR_SIZES, overwrite=True)

        def _ingest_embedding(embedding_model: str):
            model_embedding_dir = self.embedding_dir / f"{embedding_model}"
            if not model_embedding_dir.exists() or not model_embedding_dir.is_dir():
                model_embedding_dir = self.embedding_dir
            if not model_embedding_dir.exists() or not model_embedding_dir.is_dir():
                logger.warning(f"Embedding directory {model_embedding_dir} not found. Skipping {embedding_model}.")
                return

            points = []
            ingested_count = 0

            def _upsert_points(batch: List[PointStruct]):
                nonlocal ingested_count
                if not batch:
                    return

                for attempt in range(1, UPSERT_MAX_RETRIES + 1):
                    try:
                        self.qdrant_client.upsert(
                            collection_name=self.qdrant_collection_name,
                            points=batch,
                            wait=True,
                            timeout=UPSERT_TIMEOUT_SECONDS,
                        )
                        ingested_count += len(batch)
                        if ingested_count % INGEST_PROGRESS_INTERVAL < len(batch):
                            logger.info(f"{embedding_model}: ingested {ingested_count:,} points...")
                        return
                    except Exception as e:
                        if attempt == UPSERT_MAX_RETRIES:
                            raise

                        sleep_seconds = min(2 ** attempt, 30)
                        logger.warning(
                            f"{embedding_model}: upsert batch failed on attempt "
                            f"{attempt}/{UPSERT_MAX_RETRIES}: {e}. Retrying in {sleep_seconds}s..."
                        )
                        time.sleep(sleep_seconds)

            # Directory structure: embedding_dir / model_name / video_id / keyframe_<idx>.pt
            for video_id in os.listdir(model_embedding_dir):
                video_dir = model_embedding_dir / video_id
                if not video_dir.is_dir():
                    continue
                    
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
                        # Assuming the tensor can be squeezed to 1D array of embedding_dim
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
                    
                    if len(points) >= INGEST_BATCH_SIZE:
                        _upsert_points(points)
                        points = []
            
            # Upsert any remaining points
            _upsert_points(points)
            logger.info(f"Ingested {ingested_count:,} embeddings for model '{embedding_model}' into Qdrant.")

        def _ingest_object_detection():
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
                    logger.info(f"Batched object payload update: {count} points processed so far...")
                except Exception as e:
                    logger.error(f"Failed batch update for object detection payloads: {e}")
                operations = []

            for root, _, files in os.walk(obj_dir):
                for file_name in files:
                    if not file_name.endswith(".json"):
                        continue
                    
                    json_path = Path(root) / file_name
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
        
        def _ingest_ocr():
            pass
        
        for model_name in VECTOR_SIZES.keys():    
            _ingest_embedding(model_name)
        _ingest_object_detection()
        _ingest_ocr()

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

    def semantic_search(self, query: str, model_names: list = None, objects: list = None, top_k: int = 1000, score_threshold: float = 0.0, group_by_shot: bool = False, limit: int = 100):
        logger.info(f"Performing semantic search for: '{query}' with model(s): {model_names}.")
        
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

    def temporal_search(self, queries: list, model_names: list = None, objects: list = None, group_by_shot: bool = False, score_threshold: float = 0.3, limit: int = 100):
        logger.info(f"Performing temporal search for {len(queries)} queries with group_by_shot={group_by_shot}")
        
        # 1. Search each query
        query_results = []
        for q in queries:
            # We get more than 1000 to ensure we have enough paths, or keep it 1000
            res = self.semantic_search(q, model_names=model_names, objects=objects, top_k=2000, score_threshold=score_threshold, group_by_shot=group_by_shot)
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
