from data_processor.text_encoder import TextEncoder
import logging
import sys

# Configure logging to output to both console and a file called 'app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

import os
import torch
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
# from elasticsearch import Elasticsearch

from data_processor.text_encoder import *

from config import DATA_DIR, QDRANT_HOST_URL, ES_HOST_URL, QDRANT_COLLECTION_NAME, VECTOR_SIZES, EMBEDDING_WEIGHTS
from utils import setup_qdrant_collection
import json
import bisect

class RetrievalSystem:
    def __init__(self, data_dir: str=DATA_DIR, device=None, re_ingest: bool=False):
        logger.info("Initializing Video Retrieval System...")

        # auto use gpu
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.data_dir = Path(data_dir)
        self.embedding_dir = self.data_dir / "embedding"

        self.qdrant_client = QdrantClient(url=QDRANT_HOST_URL)
        self.qdrant_collection_name = QDRANT_COLLECTION_NAME
        if not self.qdrant_client.collection_exists(self.qdrant_collection_name):
            logger.warning(f"Qdrant collection '{self.qdrant_collection_name}' does not exist. Please run ingestion.")

        # self.es_client = Elasticsearch(ES_HOST_URL, request_timeout=30)

        if re_ingest:
            self.ingest()

        self.text_encoders = {}
        logger.info("Initializing text encoders...")
        for model_name in VECTOR_SIZES.keys():
            if model_name == "CLIP_H14":
                self.text_encoders[model_name] = CLIPTextEncoder(device=self.device)

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

    def ingest(self):
        setup_qdrant_collection(self.qdrant_client, self.qdrant_collection_name, VECTOR_SIZES, overwrite=True)

        def _ingest_embedding(embedding_model: str):
            model_embedding_dir = self.embedding_dir / f"{embedding_model}"
            if not model_embedding_dir.exists() or not model_embedding_dir.is_dir():
                logger.warning(f"Embedding directory {model_embedding_dir} not found. Skipping {embedding_model}.")
                return

            points = []
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
                        
                    # Load torch tensor and convert to python list
                    try:
                        # Assuming the tensor can be squeezed to 1D array of embedding_dim
                        embedding_tensor = torch.load(video_dir / file_name, map_location="cpu", weights_only=True)
                        embedding = embedding_tensor.squeeze().tolist()
                    except Exception as e:
                        logger.error(f"Failed to load {file_name} for video {video_id}: {e}")
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
                        points = []
            
            # Upsert any remaining points
            if points:
                self.qdrant_client.upsert(
                    collection_name=self.qdrant_collection_name,
                    points=points
                )
            logger.info(f"Ingested embeddings for model '{embedding_model}' into Qdrant.")

        def _ingest_object_detection():
            pass
        
        def _ingest_ocr():
            pass
        
        for model_name in VECTOR_SIZES.keys():    
            _ingest_embedding(model_name)
        _ingest_object_detection()
        _ingest_ocr()

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

    def semantic_search(self, query: str, model_names: list = None, top_k: int = 1000):
        logger.info(f"Performing semantic search for: '{query}' with model(s): {model_names}.")
        
        # Encode query
        encoded_vectors = self.encode_query(query, model_names)
        
        #Search Qdrant
        normalized_results = {} # dict mapping model_name -> dict(point_id -> normalized_score)
        all_payloads = {}       # dict mapping point_id -> payload to recover video_id and keyframe_idx later
        
        for model_name, vector in encoded_vectors.items():
            if model_name not in EMBEDDING_WEIGHTS:
                logger.warning(f"Skipping '{model_name}' search because it has no weight in EMBEDDING_WEIGHTS.")
                continue
                
            # Extract 1D array since encode_query returns shape (1, dim)
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
        for model_name, scores_dict in normalized_results.items():
            weight = EMBEDDING_WEIGHTS.get(model_name)
            for point_id, norm_score in scores_dict.items():
                if point_id not in final_scores:
                    final_scores[point_id] = 0.0
                final_scores[point_id] += norm_score * weight
                
        # Rank and return
        ranked_points = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for point_id, total_score in ranked_points:
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
            
        return results

if __name__ == "__main__":
    system = RetrievalSystem(re_ingest=True)
    results = system.semantic_search("A person riding a horse on a beach.")