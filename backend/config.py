DATA_DIR = "data"

QDRANT_HOST_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "video_frames"

ES_HOST_URL = "http://localhost:9200"

VECTOR_SIZES = {
        "CLIP_H14": 1024,
        "SigLIP": 1152,
    }

EMBEDDING_WEIGHTS = {
        "CLIP_H14": 1.0,
        "SigLIP": 1.0,
    }

MAX_FRAME_GAP = 2000

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
CACHE_TTL = 3600  # 1 hour in seconds