import os
import uuid
import torch
import logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from backend.utils import setup_qdrant_collection

logger = logging.getLogger(__name__)

def ingest_embeddings(qdrant_client: QdrantClient, collection_name: str, embedding_dir: Path, vector_sizes: dict):
    """Creates a Qdrant collection and ingests all pre-computed embedding torch files in batches."""
    setup_qdrant_collection(qdrant_client, collection_name, vector_sizes, overwrite=True)

    def _ingest_embedding(embedding_model: str):
        model_embedding_dir = embedding_dir / f"{embedding_model}"
        if not model_embedding_dir.exists() or not model_embedding_dir.is_dir():
            logger.warning(f"Embedding directory {model_embedding_dir} not found. Skipping {embedding_model}.")
            return

        points = []
        for video_id in os.listdir(model_embedding_dir):
            video_dir = model_embedding_dir / video_id
            if not video_dir.is_dir():
                continue
                
            for file_name in os.listdir(video_dir):
                if not file_name.endswith(".pt") or not file_name.startswith("keyframe_"):
                    continue
                
                try:
                    keyframe_idx = int(file_name.split("_")[1].split(".")[0])
                except (IndexError, ValueError):
                    logger.warning(f"Skipping file with unexpected name format: {file_name}")
                    continue
                    
                try:
                    embedding_tensor = torch.load(video_dir / file_name, map_location="cpu", weights_only=True)
                    embedding = embedding_tensor.squeeze().tolist()
                except Exception as e:
                    logger.error(f"Failed to load {file_name} for video {video_id}: {e}")
                    continue
                
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{keyframe_idx}"))
                
                points.append(PointStruct(
                    id=point_id,
                    vector={embedding_model: embedding},
                    payload={
                        "video_id": video_id,
                        "keyframe_idx": keyframe_idx
                    }
                ))
                
                if len(points) >= 1000:
                    qdrant_client.upsert(collection_name=collection_name, points=points)
                    points = []
        
        if points:
            qdrant_client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Ingested embeddings for model '{embedding_model}' into Qdrant.")

    def _ingest_object_detection():
        pass
    
    def _ingest_ocr():
        pass
    
    for model_name in vector_sizes.keys():
        _ingest_embedding(model_name)
    _ingest_object_detection()
    _ingest_ocr()
