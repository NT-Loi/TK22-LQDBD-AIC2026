from data_processor import config
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

from config import DATA_DIR, QDRANT_HOST_URL, ES_HOST_URL, QDRANT_COLLECTION_NAME, VECTOR_SIZES
from utils import setup_qdrant_collection

class VideoRetrievalSystem:
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

    def process_data(self):
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
                    
                    # Extract frame_idx from "keyframe_<idx>.pt"
                    try:
                        frame_idx = int(file_name.split("_")[1].split(".")[0])
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
                    
                    # Deterministic UUID based on video_id and frame_idx
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{frame_idx}"))
                    
                    points.append(PointStruct(
                        id=point_id,
                        vector={embedding_model: embedding}, # Named vector
                        payload={
                            "video_id": video_id,
                            "frame_idx": frame_idx
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

    def encode_text(self, model_name: str):
        pass

if __name__ == "__main__":
    system = VideoRetrievalSystem(re_ingest=True)