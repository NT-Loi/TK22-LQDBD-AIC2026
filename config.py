DATA_DIR = "data"

QDRANT_HOST_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "video_frames"

ES_HOST_URL = "http://localhost:9200"
ES_INDEX_NAME = "ocr_frames"

OCR_SOURCES = ["V6", "VL1.6"]

ES_TRANSCRIPT_INDEX_NAME = "transcript_segments"
CAPTION_DIR = "data/caption"
ES_CAPTION_INDEX_NAME = "shot_captions"

VECTOR_SIZES = {
        # "CLIP_H14": 1024,
        "SigLIP": 1152,
        "SigLIP2": 1536,
        "Qwen3_VL": 2048,
    }

EMBEDDING_WEIGHTS = {
        # "CLIP_H14": 1.0,
        "SigLIP": 1.0,
        "SigLIP2": 1.0,
        "Qwen3_VL": 1.0,
    }

MAX_FRAME_GAP = 2000