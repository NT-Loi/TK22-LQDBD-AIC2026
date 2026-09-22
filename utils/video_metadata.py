import os
import json
import cv2
import logging

logger = logging.getLogger(__name__)

METADATA_PATH = "data/video_metadata.json"
VIDEO_DIR = "data/video"

SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

def generate_video_metadata(force_rescan: bool = False):
    """Scans the video directory to extract FPS for each video (.mp4, .mov, etc.) and saves it to a JSON file."""
    if not os.path.exists(VIDEO_DIR):
        logger.warning(f"Video directory {VIDEO_DIR} does not exist.")
        return {}
        
    metadata = {}
    if not force_rescan and os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read existing {METADATA_PATH}, will rescan all: {e}")
            metadata = {}

    logger.info(f"Scanning videos in {VIDEO_DIR} to generate/update metadata...")
    scanned_count = 0
    
    for filename in sorted(os.listdir(VIDEO_DIR)):
        if filename.lower().endswith(SUPPORTED_VIDEO_EXTS):
            video_id = os.path.splitext(filename)[0]
            if video_id in metadata and not force_rescan:
                continue

            video_path = os.path.join(VIDEO_DIR, filename)
            scanned_count += 1
            
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    fps_val = round(fps, 3)
                    if abs(fps_val - round(fps_val)) < 0.005:
                        fps_val = float(round(fps_val))
                    metadata[video_id] = {"fps": fps_val}
                else:
                    metadata[video_id] = {"fps": 25.0} # Fallback
                cap.release()
            else:
                metadata[video_id] = {"fps": 25.0} # Fallback
                
    # Ensure data directory exists
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Successfully generated/updated metadata. Total videos: {len(metadata)} (newly scanned: {scanned_count}).")
    return metadata

def load_video_metadata():
    """Loads video metadata from JSON, generating it if it doesn't exist."""
    if not os.path.exists(METADATA_PATH):
        logger.info(f"Metadata file {METADATA_PATH} not found. Generating...")
        return generate_video_metadata()
        
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_video_metadata()
