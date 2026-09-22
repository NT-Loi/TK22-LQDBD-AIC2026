DATA_DIR = "data"

QDRANT_HOST_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "video_frames"
QDRANT_SHOT_CAPTION_COLLECTION_NAME = "shot_captions"

ES_HOST_URL = "http://localhost:9200"
ES_OCR_INDEX_NAME = "ocr_frames"
ES_TRANSCRIPT_INDEX_NAME = "transcript_segments"
ES_CAPTION_INDEX_NAME = "shot_captions"

OCR_SOURCES = ["V6", "VL1.6"]

VISION_EMBEDDING_DIM = {
        # "CLIP_H14": 1024,
        # "SigLIP": 1152,
        "SigLIP2": 1536,
        "Qwen3_VL_Embedding": 2048,
        "FG_CLIP2": 1152
    }

EMBEDDING_WEIGHTS = {
        "CLIP_H14": 1.0,
        "SigLIP": 1.0,
        "SigLIP2": 1.0,
        "Qwen3_VL_Embedding": 1.0,
        "FG_CLIP2": 1.0,
    }

CAPTION_ASPECT_KEYS = [
    "chu_the_hanh_dong",   # Subject & Action
    "vat_the_dac_diem",    # Objects & Visual Attributes
    "boi_canh_khong_gian", # Scene & Environment
    "chu_logo_man_hinh",   # Text, Logos, Screen
    "goc_nhin_co_canh",    # Camera angle, shot size, camera motion
    "dien_bien_thoi_gian", # Temporal progression
    "tong_the_canh_quay",  # Overall summary
    "full_caption",        # Complete raw caption text
]

CAPTION_EMBEDDING_DIM = {aspect: 1024 for aspect in CAPTION_ASPECT_KEYS}

MAX_FRAME_GAP = 2000