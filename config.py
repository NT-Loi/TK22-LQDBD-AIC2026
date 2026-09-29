import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")

# Secondary data directories (only loaded if EXTRA_DATA_DIRS is defined in .env)
_EXTRA_DATA_DIRS_ENV = os.getenv("EXTRA_DATA_DIRS")
EXTRA_DATA_DIRS = [
    Path(p.strip()) for p in _EXTRA_DATA_DIRS_ENV.split(";") if p.strip()
] if _EXTRA_DATA_DIRS_ENV else []

# All video source directories (searched in order)
VIDEO_DIRS = [
    Path(DATA_DIR) / "video",
    *[Path(extra) / "video" for extra in EXTRA_DATA_DIRS if (Path(extra) / "video").is_dir()],
]

# All keyframe source directories
KEYFRAME_DIRS = [
    Path(DATA_DIR) / "keyframe",
    *[Path(extra) / "keyframe" for extra in EXTRA_DATA_DIRS if (Path(extra) / "keyframe").is_dir()],
]

# All vision embedding source directories
EMBEDDING_DIRS = [
    Path(DATA_DIR) / "embedding",
    *[Path(extra) / "embedding" for extra in EXTRA_DATA_DIRS if (Path(extra) / "embedding").is_dir()],
]

# All maps source directories
MAP_DIRS = [
    Path(DATA_DIR) / "maps",
    *[Path(extra) / "maps" for extra in EXTRA_DATA_DIRS if (Path(extra) / "maps").is_dir()],
]


QDRANT_HOST_URL = os.getenv("QDRANT_HOST_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "video_frames")
QDRANT_SHOT_CAPTION_COLLECTION_NAME = os.getenv("QDRANT_SHOT_CAPTION_COLLECTION_NAME", "shot_captions")

ES_HOST_URL = os.getenv("ES_HOST_URL", "http://localhost:9200")
# ES_HOST_URL = "http://172.27.96.1:9200"
ES_OCR_INDEX_NAME = os.getenv("ES_OCR_INDEX_NAME", "ocr_frames")
ES_TRANSCRIPT_INDEX_NAME = os.getenv("ES_TRANSCRIPT_INDEX_NAME", "transcript_segments")
ES_CAPTION_INDEX_NAME = os.getenv("ES_CAPTION_INDEX_NAME", "shot_captions")

OCR_SOURCES = ["V6", "VL1.6"]

VISION_MODELS = {
    "CLIP":               {"dim": 512,  "weight": 1.0},
    "CLIP_H14":           {"dim": 1024, "weight": 1.0},
    "SigLIP":             {"dim": 1152, "weight": 1.0},
    "SigLIP2":            {"dim": 1536, "weight": 1.0},
    "Qwen3_VL_Embedding": {"dim": 2048, "weight": 1.0},
    "FG_CLIP2":           {"dim": 1152, "weight": 1.0},
}

CAPTION_MODEL = "Qwen3_Embedding"

VISION_EMBEDDING_DIM = {k: v["dim"] for k, v in VISION_MODELS.items()}
EMBEDDING_WEIGHTS    = {k: v["weight"] for k, v in VISION_MODELS.items()}

# Default vision model for interactive search (ultra-fast ~200ms)
DEFAULT_VISION_MODEL = "SigLIP2"
HNSW_EF_SEARCH = 64


# Which text encoders to load at init
# TEXT_ENCODERS = ["SigLIP2", CAPTION_MODEL, "CLIP", "Qwen3_VL_Embedding"]
# TEXT_ENCODERS = ["SigLIP2", CAPTION_MODEL, "CLIP"]
TEXT_ENCODERS = ["SigLIP2", CAPTION_MODEL, "Qwen3_VL_Embedding"]

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

# Fusion weights for combined keyframe + caption search (must sum to 1.0)
KEYFRAME_SEARCH_WEIGHT = 0.5
CAPTION_SEARCH_WEIGHT = 0.5

# Hybrid weights for shot caption search (dense semantic vs lexical BM25, must sum to 1.0)
CAPTION_DENSE_WEIGHT = 0.6
CAPTION_BM25_WEIGHT = 0.4

# Auto-translate English query to Vietnamese for shot caption search
AUTO_TRANSLATE_EN_CAPTION = False