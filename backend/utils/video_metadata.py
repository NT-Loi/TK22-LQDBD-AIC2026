import os
import json
import cv2
import logging

logger = logging.getLogger(__name__)

METADATA_PATH = "data/video_metadata.json"
VIDEO_DIR = "data/video"

def generate_video_metadata():
    """Scans the video directory to extract FPS for each video and saves it to a JSON file."""
    if not os.path.exists(VIDEO_DIR):
        logger.warning(f"Video directory {VIDEO_DIR} does not exist.")
        return {}
        
    metadata = {}
    logger.info(f"Scanning videos in {VIDEO_DIR} to generate metadata...")
    
    for filename in os.listdir(VIDEO_DIR):
        if filename.endswith(".mp4"):
            video_id = filename.split(".")[0]
            video_path = os.path.join(VIDEO_DIR, filename)
            
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    metadata[video_id] = {"fps": fps}
                else:
                    metadata[video_id] = {"fps": 25.0} # Fallback
                cap.release()
            else:
                metadata[video_id] = {"fps": 25.0} # Fallback
                
    # Ensure data directory exists
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Successfully generated metadata for {len(metadata)} videos.")
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
