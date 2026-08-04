import json
import logging
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger(__name__)

METADATA_PATH = "data/video_metadata.json"
VIDEO_DIR = "data/video"


def video_url_from_relative_path(relative_path: str) -> str:
    parts = Path(relative_path).parts
    return "/video/" + "/".join(quote(part) for part in parts)


def discover_video_files(video_root: Path) -> dict[str, Path]:
    videos = {}
    duplicates = {}
    for video_path in sorted(video_root.rglob("*.mp4")):
        video_id = video_path.stem
        if video_id in videos:
            duplicates.setdefault(video_id, []).append(video_path)
            continue
        videos[video_id] = video_path

    if duplicates:
        examples = ", ".join(sorted(duplicates)[:5])
        logger.warning(
            "Found %s duplicated video_id values under %s. Keeping the first path for each. Examples: %s",
            len(duplicates),
            video_root,
            examples,
        )
    return videos


def metadata_matches_video_tree(metadata: dict, video_root: Path) -> bool:
    videos = discover_video_files(video_root)
    if set(metadata) != set(videos):
        return False

    for video_id, video_path in videos.items():
        item = metadata.get(video_id, {})
        relative_path = item.get("path")
        if not relative_path:
            return False
        if (video_root / relative_path).resolve() != video_path.resolve():
            return False
        if item.get("url") != video_url_from_relative_path(relative_path):
            return False

    return True


def generate_video_metadata():
    """Scans the video directory to extract FPS for each video and saves it to a JSON file."""
    video_root = Path(VIDEO_DIR)
    if not video_root.exists():
        logger.warning(f"Video directory {VIDEO_DIR} does not exist.")
        return {}
        
    metadata = {}
    logger.info(f"Scanning videos in {VIDEO_DIR} to generate metadata...")
    try:
        import cv2
    except ImportError:
        cv2 = None
        logger.warning("OpenCV is not installed. Falling back to 25 FPS metadata.")
    
    for video_id, video_path in discover_video_files(video_root).items():
        relative_path = video_path.relative_to(video_root).as_posix()

        if cv2 is None:
            fps = 0
            frame_count = 0
        else:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
            else:
                fps = 0
                frame_count = 0

        metadata[video_id] = {
            "fps": fps if fps and fps > 0 else 25.0,
            "frame_count": int(frame_count) if frame_count and frame_count > 0 else None,
            "path": relative_path,
            "url": video_url_from_relative_path(relative_path),
        }
                
    Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Successfully generated metadata for {len(metadata)} videos.")
    return metadata

def load_video_metadata():
    """Loads video metadata from JSON, generating it if it doesn't exist."""
    if not Path(METADATA_PATH).exists():
        logger.info(f"Metadata file {METADATA_PATH} not found. Generating...")
        return generate_video_metadata()
        
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    video_root = Path(VIDEO_DIR)
    if not metadata or not metadata_matches_video_tree(metadata, video_root):
        logger.info("Video metadata is empty or outdated. Regenerating...")
        return generate_video_metadata()

    return metadata

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_video_metadata()
