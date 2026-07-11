import os
import json
import bisect
import torch
import logging
import numpy as np
import redis
from pathlib import Path
from qdrant_client import QdrantClient

from backend.data_processor.text_encoder import CLIPTextEncoder, SigLIPTextEncoder
from backend.config import (DATA_DIR, QDRANT_HOST_URL, QDRANT_COLLECTION_NAME, 
                            VECTOR_SIZES, EMBEDDING_WEIGHTS, MAX_FRAME_GAP,
                            REDIS_HOST, REDIS_PORT, REDIS_DB, CACHE_TTL)
from .ingestion import ingest_embeddings
from .pathfinder import find_temporal_sequences

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RetrievalSystem:
    def __init__(self, data_dir: str = DATA_DIR, device = None, re_ingest: bool = False):
        logger.info("Initializing Video Retrieval System...")

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.data_dir = Path(data_dir)
        self.embedding_dir = self.data_dir / "embedding"

        self.qdrant_client = QdrantClient(url=QDRANT_HOST_URL)
        self.qdrant_collection_name = QDRANT_COLLECTION_NAME
        
        if not self.qdrant_client.collection_exists(self.qdrant_collection_name):
            logger.warning(f"Qdrant collection '{self.qdrant_collection_name}' does not exist. Please run ingestion.")

        if re_ingest:
            self.ingest()

        self.text_encoders = {}
        logger.info("Initializing text encoders...")
        for model_name in VECTOR_SIZES.keys():
            if model_name == "CLIP_H14":
                self.text_encoders[model_name] = CLIPTextEncoder(device=self.device)
            elif model_name == "SigLIP":
                self.text_encoders[model_name] = SigLIPTextEncoder(device=self.device)

        self.shots_data = {}
        self._load_shots()

        # Initialize Redis client with graceful fallback
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            logger.info("Connected to Redis successfully. Caching is enabled.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Running without cache.")
            self.redis_client = None
            self.redis_enabled = False

    def _get_cache(self, key: str):
        if not self.redis_enabled or not self.redis_client:
            return None
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
            return None

    def _set_cache(self, key: str, value: str, ttl: int = CACHE_TTL):
        if not self.redis_enabled or not self.redis_client:
            return
        try:
            self.redis_client.set(key, value, ex=ttl)
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")

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
                            self.shots_data[video_id] = sorted(boundaries, key=lambda x: x[0])
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")

    def process_video_data(self):
        pass

    def ingest(self):
        logger.info("Executing Qdrant database ingestion...")
        ingest_embeddings(
            self.qdrant_client, 
            self.qdrant_collection_name, 
            self.embedding_dir, 
            VECTOR_SIZES
        )

    def encode_query(self, query: str, model_names: list = None):
        if model_names is None:
            model_names = list(self.text_encoders.keys())
            
        encoded_vectors = {}
        for model_name in model_names:
            if model_name not in self.text_encoders:
                logger.warning(f"Text encoder for '{model_name}' is not initialized.")
                continue
                
            cache_key = f"emb:{model_name}:{query}"
            cached_val = self._get_cache(cache_key)
            if cached_val is not None:
                try:
                    vector_list = json.loads(cached_val)
                    encoded_vectors[model_name] = np.array(vector_list, dtype=np.float32)
                    logger.debug(f"Cache hit for embedding: {model_name} -> '{query}'")
                    continue
                except Exception as e:
                    logger.warning(f"Error loading cached embedding: {e}")
                    
            # Cache miss, encode and cache the embedding
            vector = self.text_encoders[model_name](query)
            encoded_vectors[model_name] = vector
            self._set_cache(cache_key, json.dumps(vector.tolist()))
                
        return encoded_vectors

    def semantic_search(self, query: str, model_names: list = None, top_k: int = 1000, 
                        score_threshold: float = 0.0, group_by_shot: bool = False, limit: int = 100):
        logger.info(f"Performing semantic search for: '{query}' with model(s): {model_names}.")
        
        # Check cache first
        models_str = ",".join(sorted(model_names)) if model_names else "all"
        cache_key = f"search:semantic:{query}:{models_str}:{top_k}:{score_threshold}:{int(group_by_shot)}:{limit}"
        cached_res = self._get_cache(cache_key)
        if cached_res is not None:
            logger.info(f"Cache hit for semantic search: '{query}'")
            try:
                return json.loads(cached_res)
            except Exception as e:
                logger.warning(f"Failed to parse cached semantic search results: {e}")

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
            
            if not search_result:
                continue
                
            scores = [hit.score for hit in search_result]
            min_score = min(scores)
            max_score = max(scores)
            
            model_normalized_scores = {}
            for hit in search_result:
                all_payloads[hit.id] = hit.payload
                
                if max_score == min_score:
                    norm_score = 1.0 if max_score > 0 else 0.0
                else:
                    norm_score = (hit.score - min_score) / (max_score - min_score)
                    
                model_normalized_scores[hit.id] = norm_score
                
            normalized_results[model_name] = model_normalized_scores
            
        final_scores = {}
        total_weight = sum(EMBEDDING_WEIGHTS.values())
        for model_name, scores_dict in normalized_results.items():
            weight = EMBEDDING_WEIGHTS.get(model_name)
            for point_id, norm_score in scores_dict.items():
                if point_id not in final_scores:
                    final_scores[point_id] = 0.0
                final_scores[point_id] += float(norm_score * weight / total_weight)
                
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
                boundaries = self.shots_data[video_id]
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
            final_res = grouped_results[:limit]
        else:
            final_res = results[:limit]
            
        self._set_cache(cache_key, json.dumps(final_res))
        return final_res

    def temporal_search(self, queries: list, model_names: list = None, group_by_shot: bool = False, 
                        score_threshold: float = 0.3, limit: int = 100):
        logger.info(f"Performing temporal search for {len(queries)} queries with group_by_shot={group_by_shot}")
        
        # Check cache first
        queries_str = "||".join(queries)
        models_str = ",".join(sorted(model_names)) if model_names else "all"
        cache_key = f"search:temporal:{queries_str}:{models_str}:{int(group_by_shot)}:{score_threshold}:{limit}"
        cached_res = self._get_cache(cache_key)
        if cached_res is not None:
            logger.info(f"Cache hit for temporal search: {queries}")
            try:
                return json.loads(cached_res)
            except Exception as e:
                logger.warning(f"Failed to parse cached temporal search results: {e}")

        query_results = []
        for q in queries:
            res = self.semantic_search(
                q, 
                model_names=model_names, 
                top_k=2000, 
                score_threshold=score_threshold, 
                group_by_shot=group_by_shot
            )
            query_results.append(res)
            
        final_res = find_temporal_sequences(
            query_results, 
            queries, 
            group_by_shot=group_by_shot, 
            max_frame_gap=MAX_FRAME_GAP, 
            limit=limit
        )
        self._set_cache(cache_key, json.dumps(final_res))
        return final_res
