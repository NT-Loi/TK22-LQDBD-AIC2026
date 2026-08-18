DATA_DIR = "data"

QDRANT_HOST_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "video_frames"

ES_HOST_URL = "http://localhost:9200"
ES_INDEX_NAME = "ocr_frames"

OCR_SOURCES = ["PaddleOCR-VL-1.6", "PP-OCRv6"]

ES_TRANSCRIPT_INDEX_NAME = "transcript_segments"

VECTOR_SIZES = {
        # "CLIP_H14": 1024,
        "SigLIP": 1152,
        "SigLIP2": 1536,
    }

EMBEDDING_WEIGHTS = {
        # "CLIP_H14": 1.0,
        "SigLIP": 1.0,
        "SigLIP2": 1.0,
    }

MAX_FRAME_GAP = 2000