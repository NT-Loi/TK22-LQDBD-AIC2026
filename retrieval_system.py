import logging
import sys

# Configure logging to output to both console and a file called 'app.log'
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

import os
import torch
import uuid
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchAny, Range, SetPayloadOperation, SetPayload, SearchParams
from elasticsearch import Elasticsearch

from data_processor.text_encoder import *

from config import *
from utils import *
from utils.video_metadata import load_video_metadata
import json
import bisect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RetrievalSystem:
    def __init__(self, data_dir: str=DATA_DIR, device=None, re_ingest: bool=False, init_text_encoders: bool=True):
        logger.info("Initializing Video Retrieval System...")

        # auto use gpu
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.data_dir = Path(data_dir)
        self.vision_embedding_dir = self.data_dir / "embedding"
        self.caption_dir = self.data_dir / "caption"
        self.caption_embedding_dir = self.data_dir / "caption_embedding"
        self.shot_dir = self.data_dir / "shot"
        self.obj_dir = self.data_dir / "object_detection"
        self.ocr_dir = self.data_dir / "ocr"
        self.transcript_dir = self.data_dir / "transcript"

        self.qdrant_client = QdrantClient(url=QDRANT_HOST_URL, prefer_grpc=True, timeout=60.0)
        
        if not self.qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
            logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' does not exist. Please run ingestion.")

        if not self.qdrant_client.collection_exists(QDRANT_SHOT_CAPTION_COLLECTION_NAME):
            logger.warning(f"Qdrant collection '{QDRANT_SHOT_CAPTION_COLLECTION_NAME}' does not exist. Please run ingestion.")

        self.es_client = Elasticsearch(ES_HOST_URL, request_timeout=300)

        if not self.es_client.indices.exists(index=ES_CAPTION_INDEX_NAME):
            logger.warning(f"Elasticsearch index '{ES_CAPTION_INDEX_NAME}' does not exist. Please run caption ES ingestion.")

        # Load video metadata (FPS) for frame-to-time conversion
        self.video_metadata = load_video_metadata()

        if re_ingest:
            self.ingest()

        self.text_encoders = {}
        if init_text_encoders:
            logger.info("Initializing text encoders...")
            for model_name in TEXT_ENCODERS:
                if model_name == "CLIP_H14":
                    self.text_encoders[model_name] = CLIPTextEncoder(device=self.device)

                if model_name == "SigLIP":
                    self.text_encoders[model_name] = SigLIPTextEncoder(device=self.device)

                if model_name == "SigLIP2":
                    self.text_encoders[model_name] = SigLIP2TextEncoder(device=self.device)

                if model_name == "Qwen3_VL_Embedding":
                    self.text_encoders[model_name] = Qwen3VLEmbeddingTextEncoder(device=self.device)

                if model_name == "FG_CLIP2":
                    self.text_encoders[model_name] = FGCLIP2TextEncoder(device=self.device)

                if model_name == "Qwen3_Embedding":
                    self.text_encoders[model_name] = Qwen3EmbeddingTextEncoder(device=self.device)

        # Pre-load shot boundaries on RAM to efficiently group frame by
        self.shots_data = {}
        self._load_shots()

    def _load_shots(self):
        if not self.shot_dir.exists() or not self.shot_dir.is_dir():
            logger.warning(f"Shot directory {self.shot_dir} does not exist.")
            return
            
        logger.info("Loading shot boundaries...")
        for filename in os.listdir(self.shot_dir):
            if filename.endswith(".json"):
                try:
                    with open(self.shot_dir / filename, "r") as f:
                        data = json.load(f)
                        for video_key, boundaries in data.items():
                            video_id = video_key.split(".")[0]
                            # Sort just in case
                            self.shots_data[video_id] = sorted(boundaries, key=lambda x: x[0])
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")

    def process_video_data(self):
        pass

    def ingest_vision_embedding(self, embedding_model: str, max_workers: int = 8):
        model_embedding_dir = self.vision_embedding_dir / f"{embedding_model}"
        if not model_embedding_dir.exists() or not model_embedding_dir.is_dir():
            logger.warning(f"Embedding directory {model_embedding_dir} not found. Skipping {embedding_model}.")
            return

        expected_dim = VISION_EMBEDDING_DIM.get(embedding_model)
        video_dirs = [d for d in model_embedding_dir.iterdir() if d.is_dir()]

        def _process_video_dir(video_dir):
            video_id = video_dir.name
            points = []

            for file_name in os.listdir(video_dir):
                if not file_name.endswith(".pt") or not file_name.startswith("keyframe_"):
                    continue

                try:
                    keyframe_idx = int(file_name.split("_")[1].split(".")[0])
                except (IndexError, ValueError):
                    continue

                try:
                    embedding_tensor = torch.load(video_dir / file_name, map_location="cpu", weights_only=True)
                    embedding = embedding_tensor.squeeze().tolist()
                except Exception as e:
                    logger.error(f"Failed to load {file_name} for video {video_id}: {e}")
                    continue

                if len(embedding) != expected_dim:
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

            if not points:
                return 0

            return batch_upsert_points(
                self.qdrant_client,
                QDRANT_COLLECTION_NAME,
                points,
                batch_size=10000,
                wait=False,
            )

        total_ingested = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_video_dir, vd): vd for vd in video_dirs}
            for future in tqdm(as_completed(futures), total=len(video_dirs), desc=f"📦 Embeddings [{embedding_model}]", unit="video"):
                try:
                    total_ingested += future.result()
                except Exception as e:
                    vd = futures[future]
                    logger.error(f"Failed to ingest video {vd.name}: {e}")

        logger.info(f"Ingested {total_ingested} embeddings for model '{embedding_model}' into Qdrant.")

    def ingest_all_vision_embedding(self, max_workers: int = 8):
        if not self.vision_embedding_dir.exists() or not self.vision_embedding_dir.is_dir():
            logger.warning(f"Vision embedding directory {self.vision_embedding_dir} not found. Skipping.")
            return

        logger.info("Ingesting vision embedding into Qdrant...")
        setup_qdrant_collection(self.qdrant_client, QDRANT_COLLECTION_NAME, VISION_EMBEDDING_DIM, overwrite=True)
        for model_name in VISION_EMBEDDING_DIM.keys():    
            self.ingest_vision_embedding(model_name, max_workers=max_workers)

    def ingest_shot_caption_embedding(self, max_workers: int = 8):
        if not self.caption_embedding_dir.exists() or not self.caption_embedding_dir.is_dir():
            logger.warning(f"Shot caption embedding directory {self.caption_embedding_dir} not found. Skipping.")
            return

        logger.info("Ingesting shot caption embeddings into Qdrant...")
        setup_qdrant_collection(self.qdrant_client, QDRANT_SHOT_CAPTION_COLLECTION_NAME, CAPTION_EMBEDDING_DIM, overwrite=True)

        pt_files = sorted(self.caption_embedding_dir.glob("*.pt"))
        if not pt_files:
            logger.warning(f"No .pt files found in {self.caption_embedding_dir}.")
            return

        def _process_caption_file(pt_file):
            emb_data = torch.load(pt_file, map_location="cpu")

            video_id = emb_data.get("video_id", pt_file.stem)
            num_shots = emb_data.get("total_shots", 0)
            shot_indices = emb_data.get("shot_indices", list(range(num_shots)))
            start_frames = emb_data.get("start_frames", [0] * num_shots)
            end_frames = emb_data.get("end_frames", [0] * num_shots)
            rep_frames = emb_data.get("representative_frames", [0] * num_shots)
            embeddings_dict = emb_data.get("embeddings", {})

            cap_shots_map = {}
            cap_file = self.caption_dir / f"{video_id}.json"
            if cap_file.exists():
                try:
                    with open(cap_file, "r", encoding="utf-8") as jf:
                        cap_json = json.load(jf)
                        for s in cap_json.get("shots", []):
                            cap_shots_map[s["shot_idx"]] = s
                except Exception as e:
                    logger.warning(f"Could not load caption json for '{video_id}': {e}")

            points = []
            for i in range(num_shots):
                s_idx = shot_indices[i]

                vectors = {}
                for aspect in CAPTION_ASPECT_KEYS:
                    if aspect in embeddings_dict:
                        vectors[aspect] = embeddings_dict[aspect][i].tolist()

                if not vectors:
                    continue

                cap_shot = cap_shots_map.get(s_idx, {})
                payload = {
                    "video_id": video_id,
                    "shot_idx": s_idx,
                    "start_frame": start_frames[i],
                    "end_frame": end_frames[i],
                    "representative_frame": rep_frames[i],
                    "keyframe_idx": rep_frames[i],
                    "start_sec": cap_shot.get("start_sec"),
                    "end_sec": cap_shot.get("end_sec"),
                    "caption_text": cap_shot.get("caption_text", ""),
                    "aspects": cap_shot.get("aspects", {}),
                    "ocr_detected": cap_shot.get("ocr_detected", []),
                    "audio_transcript": cap_shot.get("audio_transcript", ""),
                }

                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{s_idx}"))
                points.append(PointStruct(id=point_id, vector=vectors, payload=payload))

            if not points:
                return 0

            return batch_upsert_points(
                self.qdrant_client,
                QDRANT_SHOT_CAPTION_COLLECTION_NAME,
                points,
                batch_size=200,
                wait=False,
            )

        total_ingested = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_caption_file, f): f for f in pt_files}
            for future in tqdm(as_completed(futures), total=len(pt_files), desc="📝 Caption Embeddings", unit="vid"):
                try:
                    total_ingested += future.result()
                except Exception as e:
                    pt_file = futures[future]
                    logger.error(f"Failed to ingest caption for {pt_file.name}: {e}")

        logger.info(f"Ingested {total_ingested} shot caption embeddings into Qdrant.")

    def ingest_object_detection(self):
        if not self.obj_dir.exists() or not self.obj_dir.is_dir():
            logger.warning(f"Object detection directory {self.obj_dir} not found. Skipping.")
            return

        logger.info("Ingesting object detection metadata into Qdrant...")
        count = 0
        operations = []

        def _flush_operations():
            nonlocal operations, count
            if not operations:
                return
            try:
                self.qdrant_client.batch_update_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    update_operations=operations
                )
                count += len(operations)
            except Exception as e:
                logger.error(f"Failed batch update for object detection payloads: {e}")
            operations = []

        # Collect all JSON files first for tqdm
        all_json_files = []
        for root, _, files in os.walk(self.obj_dir):
            for file_name in files:
                if file_name.endswith(".json"):
                    all_json_files.append(Path(root) / file_name)

        for json_path in tqdm(all_json_files, desc="🔍 Object Detection", unit="file"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read JSON {json_path}: {e}")
                continue
                
            image_path = data.get("image_path", "")
            if image_path:
                parent_name = Path(image_path).parent.name
                video_id = parent_name if parent_name and parent_name != "." else data.get("video_id")
            else:
                video_id = json_path.parent.name if json_path.parent.name else data.get("video_id")

            keyframe_idx = data.get("keyframe_index")
            if not video_id or keyframe_idx is None:
                continue

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{keyframe_idx}"))

            payload = {
                "objects": data.get("object_names", []),
                "has_objects": data.get("has_objects", False),
                "num_detections": data.get("num_detections", 0),
                "class_counts": data.get("class_counts", {})
            }

            operations.append(
                SetPayloadOperation(
                    set_payload=SetPayload(
                        payload=payload,
                        points=[point_id]
                    )
                )
            )

            if len(operations) >= 500:
                _flush_operations()

        _flush_operations()
        logger.info(f"Completed object detection payload ingestion for {count} points.")
    
    def ingest_ocr(self):
        if not self.ocr_dir.exists() or not self.ocr_dir.is_dir():
            logger.warning(f"OCR directory {self.ocr_dir} not found. Skipping OCR ingestion.")
            return
        setup_ocr_index(self.es_client, ES_OCR_INDEX_NAME, overwrite=True)
        ingest_ocr_to_es(self.es_client, ES_OCR_INDEX_NAME, str(self.ocr_dir), OCR_SOURCES)

    def ingest_transcript(self):
        if not self.transcript_dir.exists():
            logger.warning(f"Transcript directory {self.transcript_dir} not found. Skipping transcript ingestion.")
            return
        setup_transcript_index(self.es_client, ES_TRANSCRIPT_INDEX_NAME, overwrite=True)
        ingest_transcript_to_es(self.es_client, ES_TRANSCRIPT_INDEX_NAME, str(self.transcript_dir))

    def ingest_caption_es(self, overwrite: bool = True):
        if not self.caption_dir.exists() or not self.caption_dir.is_dir():
            logger.warning(f"Caption directory {self.caption_dir} not found. Skipping ES caption ingestion.")
            return
        setup_caption_index(self.es_client, ES_CAPTION_INDEX_NAME, overwrite=overwrite)
        ingest_caption_to_es(self.es_client, ES_CAPTION_INDEX_NAME, str(self.caption_dir), str(self.caption_embedding_dir))

    def ingest(self):
        self.ingest_all_vision_embedding()
        self.ingest_shot_caption_embedding()
        self.ingest_caption_es()
        self.ingest_object_detection()
        self.ingest_ocr()
        self.ingest_transcript()

    def encode_query(self, query: str, model_names: list = None):
        """
        Encode query using specified text encoders.
        If model_names is None, it uses all initialized text encoders.
        Returns a dictionary mapping model_name to encoded text vector.
        """
        if model_names is None:
            model_names = list(self.text_encoders.keys())
            
        encoded_vectors = {}
        for model_name in model_names:
            if model_name in self.text_encoders:
                encoded_vectors[model_name] = self.text_encoders[model_name](query)
            else:
                logger.warning(f"Text encoder for '{model_name}' is not initialized.")
                
        return encoded_vectors

    def _matches_object_filters(self, payload: dict, objects: list) -> bool:
        if not objects or not payload:
            return True
        
        frame_objects = payload.get("objects", [])
        class_counts = payload.get("class_counts", {})
        
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            label = obj.get("label")
            if not label:
                continue
                
            if label not in frame_objects:
                return False
                
            min_instances = obj.get("min_instances", 1)
            max_instances = obj.get("max_instances")
            actual_count = class_counts.get(label, 0)
            
            if actual_count < min_instances:
                return False
            if max_instances is not None and isinstance(max_instances, int) and actual_count > max_instances:
                return False
                
        return True

    def ocr_search(self, query: str, ocr_sources: list = None, fuzziness: str = "AUTO", size: int = 500):
        """
        Standalone OCR text search. Returns results in the same format as semantic_search.
        """
        logger.info(f"Performing OCR search for: '{query}'")
        ocr_hits = fuzzy_search_ocr(self.es_client, ES_OCR_INDEX_NAME, query,
                                    ocr_sources=ocr_sources, fuzziness=fuzziness, size=size)
        
        results = []
        for hit in ocr_hits:
            video_id = hit["video_id"]
            keyframe_idx = hit["keyframe_idx"]
            
            shot_start = 0
            shot_end = 0
            if video_id in self.shots_data:
                boundaries = self.shots_data[video_id]
                starts = [b[0] for b in boundaries]
                idx = bisect.bisect_right(starts, keyframe_idx) - 1
                if idx >= 0 and keyframe_idx <= boundaries[idx][1]:
                    shot_start = boundaries[idx][0]
                    shot_end = boundaries[idx][1]
            
            results.append({
                "video_id": video_id,
                "keyframe_index": keyframe_idx,
                "score": hit["es_score"],
                "shot_start_frame": shot_start,
                "shot_end_frame": shot_end
            })
        
        return results

    def _get_ocr_filter_ranges(self, ocr_query, fuzziness: str = "AUTO") -> dict:
        """
        Get OCR filter data for single or multiple OCR queries with per-query level ('frame' or 'video').
        Returns dict with 'common_vids', 'query_items', and 'combined_ocr_text_map'.
        """
        parsed_queries = []
        if isinstance(ocr_query, str):
            if ocr_query.strip():
                parsed_queries.append({"text": ocr_query.strip(), "level": "frame"})
        elif isinstance(ocr_query, list):
            for item in ocr_query:
                if isinstance(item, str) and item.strip():
                    parsed_queries.append({"text": item.strip(), "level": "frame"})
                elif isinstance(item, dict) and item.get("text", "").strip():
                    lvl = item.get("level", "frame")
                    if lvl not in ("frame", "video"):
                        lvl = "frame"
                    parsed_queries.append({"text": item["text"].strip(), "level": lvl})

        if not parsed_queries:
            return {}

        query_items = []
        combined_ocr_text_map = {}

        for q in parsed_queries:
            hits = fuzzy_search_ocr(self.es_client, ES_OCR_INDEX_NAME,
                                    q["text"], fuzziness=fuzziness, size=5000)
            vid_map = {}
            for h in hits:
                vid = h["video_id"]
                kf_idx = h["keyframe_idx"]
                if vid not in vid_map:
                    vid_map[vid] = {}
                vid_map[vid][kf_idx] = {
                    "score": h["es_score"],
                    "ocr_text": h.get("ocr_text", "")
                }
                combined_ocr_text_map[(vid, kf_idx)] = h.get("ocr_text", "")

            query_items.append({
                "text": q["text"],
                "level": q["level"],
                "vid_map": vid_map
            })

        if not query_items:
            return {}

        # AND Logic: Keep only videos present in ALL per_query_ocr_results
        common_vids = set(query_items[0]["vid_map"].keys())
        for q_item in query_items[1:]:
            common_vids &= set(q_item["vid_map"].keys())

        if not common_vids:
            return {}

        return {
            "common_vids": common_vids,
            "query_items": query_items,
            "combined_ocr_text_map": combined_ocr_text_map
        }

    def _frame_in_ocr_ranges(self, video_id: str, keyframe_idx: int, ocr_data: dict) -> float:
        """
        Check if a keyframe passes per-query OCR filters (AND logic across queries, per-query level).
        Returns combined score (>0.0) if passed, or 0.0 if failed.
        """
        if not ocr_data or "common_vids" not in ocr_data:
            return 0.0

        if video_id not in ocr_data["common_vids"]:
            return 0.0

        total_score = 0.0

        for q_item in ocr_data["query_items"]:
            level = q_item["level"]
            vid_map = q_item["vid_map"]
            frames_dict = vid_map.get(video_id, {})

            if level == "video":
                # Video level: satisfied if video matched this OCR query anywhere
                if not frames_dict:
                    return 0.0
                max_sc = max(info["score"] for info in frames_dict.values())
                total_score += max_sc
            else:
                # Frame level: keyframe itself must contain this OCR query (or within proximity window)
                if keyframe_idx in frames_dict:
                    total_score += frames_dict[keyframe_idx]["score"]
                else:
                    matched_sc = 0.0
                    for kf, info in frames_dict.items():
                        if abs(kf - keyframe_idx) <= 150: # ~5s frame window
                            matched_sc = max(matched_sc, info["score"])
                    if matched_sc > 0.0:
                        total_score += matched_sc
                    else:
                        return 0.0

        return total_score

    def _get_transcript_filter_ranges(self, audio_query, fuzziness: str = "AUTO") -> dict:
        """
        Get transcript filter data for single or multiple audio queries with per-query level ('frame' or 'video').
        Returns dict with 'common_vids' and 'query_items'.
        """
        parsed_queries = []
        if isinstance(audio_query, str):
            if audio_query.strip():
                parsed_queries.append({"text": audio_query.strip(), "level": "frame"})
        elif isinstance(audio_query, list):
            for item in audio_query:
                if isinstance(item, str) and item.strip():
                    parsed_queries.append({"text": item.strip(), "level": "frame"})
                elif isinstance(item, dict) and item.get("text", "").strip():
                    lvl = item.get("level", "frame")
                    if lvl not in ("frame", "video"):
                        lvl = "frame"
                    parsed_queries.append({"text": item["text"].strip(), "level": lvl})

        if not parsed_queries:
            return {}

        query_items = []
        for q in parsed_queries:
            hits = fuzzy_search_transcript(self.es_client, ES_TRANSCRIPT_INDEX_NAME,
                                           q["text"], fuzziness=fuzziness, size=5000)
            vid_map = {}
            for h in hits:
                vid = h["video_id"]
                if vid not in vid_map:
                    vid_map[vid] = []
                vid_map[vid].append({
                    "start": h["start"],
                    "end": h["end"],
                    "text": h.get("text", ""),
                    "score": h["es_score"]
                })
            query_items.append({
                "text": q["text"],
                "level": q["level"],
                "vid_map": vid_map
            })

        if not query_items:
            return {}

        # AND Logic: Keep only videos present in ALL per_query_results
        common_vids = set(query_items[0]["vid_map"].keys())
        for q_item in query_items[1:]:
            common_vids &= set(q_item["vid_map"].keys())

        if not common_vids:
            return {}

        return {
            "common_vids": common_vids,
            "query_items": query_items
        }

    def _get_audio_text_for_frame(self, video_id: str, keyframe_idx: int, transcript_data: dict) -> str:
        """
        Get matched audio transcript text snippet for a given keyframe.
        """
        if not transcript_data or "query_items" not in transcript_data:
            return ""

        fps = 25.0
        if video_id in self.video_metadata and "fps" in self.video_metadata[video_id]:
            fps = self.video_metadata[video_id]["fps"]

        frame_time_sec = keyframe_idx / fps
        texts = []

        for q_item in transcript_data["query_items"]:
            vid_map = q_item["vid_map"]
            ranges = vid_map.get(video_id, [])

            for r in ranges:
                s_sec, e_sec, txt = r["start"], r["end"], r.get("text", "")
                if txt and (s_sec - 2.0) <= frame_time_sec <= (e_sec + 2.0):
                    if txt not in texts:
                        texts.append(txt)
                elif txt and q_item["level"] == "video":
                    if txt not in texts:
                        texts.append(txt)

        return " | ".join(texts)

    def _frame_in_transcript_ranges(self, video_id: str, keyframe_idx: int, transcript_data: dict) -> float:
        """
        Check if a keyframe passes per-query audio filters (AND logic across queries, per-query level).
        Returns combined score (>0.0) if passed, or 0.0 if failed.
        """
        if not transcript_data or "common_vids" not in transcript_data:
            return 0.0

        if video_id not in transcript_data["common_vids"]:
            return 0.0

        fps = 25.0
        if video_id in self.video_metadata and "fps" in self.video_metadata[video_id]:
            fps = self.video_metadata[video_id]["fps"]

        frame_time_sec = keyframe_idx / fps
        total_score = 0.0

        for q_item in transcript_data["query_items"]:
            level = q_item["level"]
            vid_map = q_item["vid_map"]
            ranges = vid_map.get(video_id, [])

            if level == "video":
                # Video level: satisfied for any keyframe in this video
                scores = [r["score"] for r in ranges]
                total_score += max(scores) if scores else 1.0
            else:
                # Frame level: keyframe must be within +/- 2.0s of audio segment
                match_score = 0.0
                for r in ranges:
                    s_sec, e_sec, sc = r["start"], r["end"], r["score"]
                    if (s_sec - 2.0) <= frame_time_sec <= (e_sec + 2.0):
                        if sc > match_score:
                            match_score = sc
                if match_score == 0.0:
                    return 0.0  # Failed this frame-level requirement
                total_score += match_score

        return total_score

    def filter_search(self, ocr_query: str = None, audio_query = None, objects: list = None,
                      group_by_shot: bool = False, score_threshold: float = 0.0, limit: int = 100):
        """
        Perform search purely using metadata filters (OCR, audio transcript, object detection)
        without requiring a text query.
        """
        logger.info(f"Performing filter search with ocr_query='{ocr_query}', audio_query='{audio_query}', objects={objects}")

        if not ocr_query and not audio_query and not objects:
            return []

        # 1. Gather transcript ranges if audio_query is given
        transcript_ranges = None
        if audio_query:
            transcript_ranges = self._get_transcript_filter_ranges(audio_query)
            if not transcript_ranges and not ocr_query and not objects:
                return []

        # 2. Gather OCR ranges if ocr_query is given
        ocr_ranges = None
        if ocr_query:
            ocr_ranges = self._get_ocr_filter_ranges(ocr_query)
            if not ocr_ranges and not audio_query and not objects:
                return []

        # Map: (video_id, keyframe_idx) -> score
        candidates = {}

        if ocr_ranges:
            # Case A: OCR query provided
            for vid in ocr_ranges["common_vids"]:
                # Collect candidate frames for this video
                for q_item in ocr_ranges["query_items"]:
                    for kf_idx in q_item["vid_map"].get(vid, {}).keys():
                        ocr_sc = self._frame_in_ocr_ranges(vid, kf_idx, ocr_ranges)
                        if ocr_sc > 0.0:
                            aud_score = self._frame_in_transcript_ranges(vid, kf_idx, transcript_ranges) if transcript_ranges is not None else 1.0
                            if transcript_ranges is not None and aud_score == 0.0:
                                continue
                            candidates[(vid, kf_idx)] = ocr_sc + (aud_score if transcript_ranges is not None else 0.0)

            # Check object filter if objects present
            if objects and candidates:
                point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{vid}_{kf_idx}")) for (vid, kf_idx) in candidates.keys()]
                try:
                    retrieved_points = self.qdrant_client.retrieve(
                        collection_name=QDRANT_COLLECTION_NAME,
                        ids=point_ids,
                        with_payload=True
                    )
                    filtered_candidates = {}
                    for p in retrieved_points:
                        vid = p.payload.get("video_id")
                        kf_idx = p.payload.get("keyframe_idx", p.payload.get("frame_idx", 0))
                        if self._matches_object_filters(p.payload, objects):
                            score = candidates.get((vid, kf_idx), 1.0)
                            filtered_candidates[(vid, kf_idx)] = score
                    candidates = filtered_candidates
                except Exception as e:
                    logger.warning(f"Error batch retrieving Qdrant points: {e}")

        elif audio_query:
            # Case B: Audio query provided (no OCR query)
            qdrant_filter = build_qdrant_filter(objects) if objects else None
            
            points = []
            if qdrant_filter:
                res, _ = self.qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    scroll_filter=qdrant_filter,
                    limit=5000,
                    with_payload=True
                )
                points = res
            else:
                vids = list(transcript_ranges.get("common_vids", []))
                if vids:
                    v_filter = Filter(must=[FieldCondition(key="video_id", match=MatchAny(any=vids))])
                    res, _ = self.qdrant_client.scroll(
                        collection_name=QDRANT_COLLECTION_NAME,
                        scroll_filter=v_filter,
                        limit=5000,
                        with_payload=True
                    )
                    points = res

            for p in points:
                vid = p.payload.get("video_id")
                kf_idx = p.payload.get("keyframe_idx", p.payload.get("frame_idx", 0))
                if vid and kf_idx is not None:
                    aud_score = self._frame_in_transcript_ranges(vid, kf_idx, transcript_ranges)
                    if aud_score > 0.0:
                        candidates[(vid, kf_idx)] = aud_score

        elif objects:
            # Case C: Only object filter provided (no OCR, no audio)
            qdrant_filter = build_qdrant_filter(objects)
            points, _ = self.qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=qdrant_filter,
                limit=limit * 5 if group_by_shot else limit,
                with_payload=True
            )
            for p in points:
                vid = p.payload.get("video_id")
                kf_idx = p.payload.get("keyframe_idx", p.payload.get("frame_idx", 0))
                if vid and kf_idx is not None:
                    candidates[(vid, kf_idx)] = 1.0

        # Build formatted output results
        results = []
        ocr_text_map = ocr_ranges.get("combined_ocr_text_map", {}) if ocr_ranges else {}
        for (vid, kf_idx), score in candidates.items():
            if score_threshold > 0 and score < score_threshold and score < 1.0:
                continue

            shot_start = 0
            shot_end = 0
            if vid in self.shots_data:
                boundaries = self.shots_data[vid]
                starts = [b[0] for b in boundaries]
                idx = bisect.bisect_right(starts, kf_idx) - 1
                if idx >= 0 and kf_idx <= boundaries[idx][1]:
                    shot_start = boundaries[idx][0]
                    shot_end = boundaries[idx][1]

            results.append({
                "video_id": vid,
                "keyframe_index": kf_idx,
                "score": score,
                "shot_start_frame": shot_start,
                "shot_end_frame": shot_end,
                "ocr_text": ocr_text_map.get((vid, kf_idx), ""),
                "audio_text": self._get_audio_text_for_frame(vid, kf_idx, transcript_ranges)
            })

        if group_by_shot:
            shot_dict = {}
            for item in results:
                key = (item["video_id"], item["shot_start_frame"], item["shot_end_frame"])
                if key not in shot_dict:
                    shot_dict[key] = []
                shot_dict[key].append(item)

            grouped_results = []
            for (vid, start, end), items in shot_dict.items():
                avg_score = sum(i["score"] for i in items) / len(items)
                items.sort(key=lambda x: x["keyframe_index"])
                anchor = items[0]
                grouped_results.append({
                    "type": "shot",
                    "video_id": vid,
                    "score": avg_score,
                    "frames": items,
                    "keyframe_index": anchor["keyframe_index"],
                    "shot_start_frame": start,
                    "shot_end_frame": end
                })

            grouped_results.sort(key=lambda x: x["score"], reverse=True)
            return grouped_results[:limit]

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def semantic_search(self, query: str, model_names: list = None, objects: list = None, top_k: int = 1000, score_threshold: float = 0.0, group_by_shot: bool = False, limit: int = 100, ocr_query: str = None, audio_query = None, audio_match_level: str = "frame"):
        logger.info(f"Performing semantic search for: '{query}' with model(s): {model_names}, ocr_query: '{ocr_query}', audio_query: '{audio_query}', match_level: '{audio_match_level}'.")
        
        if not query or not query.strip():
            return self.filter_search(
                ocr_query=ocr_query,
                audio_query=audio_query,
                # audio_match_level=audio_match_level,
                objects=objects,
                group_by_shot=group_by_shot,
                score_threshold=score_threshold,
                limit=limit
            )
        
        # Build OCR filter ranges if ocr_query is provided
        ocr_ranges = None
        if ocr_query:
            ocr_ranges = self._get_ocr_filter_ranges(ocr_query)
            if not ocr_ranges:
                logger.info(f"No OCR matches for '{ocr_query}'. Returning empty results.")
                return []

        # Build transcript filter ranges if audio_query is provided
        transcript_ranges = None
        if audio_query:
            transcript_ranges = self._get_transcript_filter_ranges(audio_query)
        
        # Only encode with vision-compatible models (those with entries in VISION_MODELS)
        if model_names is None:
            vision_model_names = [m for m in self.text_encoders.keys() if m in VISION_MODELS]
        else:
            vision_model_names = [m for m in model_names if m in VISION_MODELS]

        encoded_vectors = self.encode_query(query, vision_model_names)
        
        normalized_results = {}
        all_payloads = {}
        
        for model_name, vector in encoded_vectors.items():
            if model_name not in EMBEDDING_WEIGHTS:
                logger.warning(f"Embedding weight for {model_name} hasn't been initialized. Use 1.0 as default value.")
                EMBEDDING_WEIGHTS[model_name] = 1.0

            query_vector = vector[0].tolist()
            
            search_result = self.qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=query_vector,
                using=model_name,
                limit=top_k,
                search_params=SearchParams(exact=True),
                with_payload=True
            ).points

            if objects:
                search_result = [hit for hit in search_result if self._matches_object_filters(hit.payload, objects)]
            
            if not search_result:
                continue
                
            # Min-Max Normalization
            scores = [hit.score for hit in search_result]
            min_score = min(scores)
            max_score = max(scores)
            
            model_normalized_scores = {}
            for hit in search_result:
                # Save payload for final output
                all_payloads[hit.id] = hit.payload
                
                # Handle edge case where max == min
                if max_score == min_score:
                    norm_score = 1.0 if max_score > 0 else 0.0
                else:
                    norm_score = (hit.score - min_score) / (max_score - min_score)
                    
                model_normalized_scores[hit.id] = norm_score
                
            normalized_results[model_name] = model_normalized_scores
            
        # Calculate overall score
        final_scores = {}
        total_weight = sum(EMBEDDING_WEIGHTS.values())
        for model_name, scores_dict in normalized_results.items():
            weight = EMBEDDING_WEIGHTS.get(model_name)
            for point_id, norm_score in scores_dict.items():
                if point_id not in final_scores:
                    final_scores[point_id] = 0.0
                final_scores[point_id] += float(norm_score * weight / total_weight)
                
        # Apply OCR intersection filter before ranking
        if ocr_ranges is not None:
            final_scores = {
                pid: score for pid, score in final_scores.items()
                if self._frame_in_ocr_ranges(
                    all_payloads[pid].get("video_id"),
                    all_payloads[pid].get("keyframe_idx", all_payloads[pid].get("frame_idx", 0)),
                    ocr_ranges
                ) > 0.0
            }
            logger.info(f"After OCR filter: {len(final_scores)} results remain.")

        # Apply transcript intersection filter before ranking
        if transcript_ranges is not None:
            final_scores = {
                pid: score for pid, score in final_scores.items()
                if self._frame_in_transcript_ranges(
                    all_payloads[pid].get("video_id"),
                    all_payloads[pid].get("keyframe_idx", all_payloads[pid].get("frame_idx", 0)),
                    transcript_ranges
                ) > 0.0
            }
            logger.info(f"After transcript filter: {len(final_scores)} results remain.")

        # Rank and return
        ranked_points = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        ocr_text_map = ocr_ranges.get("combined_ocr_text_map", {}) if ocr_ranges else {}
        for point_id, total_score in ranked_points:
            if total_score < score_threshold:
                continue
                
            payload = all_payloads[point_id]
            video_id = payload.get("video_id")
            keyframe_idx = payload.get("keyframe_idx", payload.get("frame_idx"))
            
            shot_start = 0
            shot_end = 0
            if video_id in self.shots_data:
                # Binary search to find the shot containing keyframe_idx
                boundaries = self.shots_data[video_id]
                # Extract just the start frames for bisect
                starts = [b[0] for b in boundaries]
                idx = bisect.bisect_right(starts, keyframe_idx) - 1
                if idx >= 0 and keyframe_idx <= boundaries[idx][1]:
                    shot_start = boundaries[idx][0]
                    shot_end = boundaries[idx][1]
            
            results.append({
                "video_id": video_id,
                "keyframe_index": keyframe_idx,
                "score": total_score,
                "shot_start_frame": shot_start,
                "shot_end_frame": shot_end,
                "ocr_text": ocr_text_map.get((video_id, keyframe_idx), ""),
                "audio_text": self._get_audio_text_for_frame(video_id, keyframe_idx, transcript_ranges)
            })
            
        if group_by_shot:
            shot_dict = {}
            for item in results:
                key = (item["video_id"], item["shot_start_frame"], item["shot_end_frame"])
                if key not in shot_dict:
                    shot_dict[key] = []
                shot_dict[key].append(item)
                
            grouped_results = []
            for (vid, start, end), items in shot_dict.items():
                avg_score = sum(i["score"] for i in items) / len(items)
                items.sort(key=lambda x: x["keyframe_index"])
                anchor = items[0]
                grouped_results.append({
                    "type": "shot",
                    "video_id": vid,
                    "score": avg_score,
                    "frames": items,
                    "keyframe_index": anchor["keyframe_index"],
                    "shot_start_frame": start,
                    "shot_end_frame": end
                })
                
            grouped_results.sort(key=lambda x: x["score"], reverse=True)
            return grouped_results[:limit]
            
        return results[:limit]

    def temporal_search(self, queries: list, model_names: list = None, objects: list = None, group_by_video: bool = False, score_threshold: float = 0.0, limit: int = 100, ocr_query = None, audio_query = None, audio_match_level: str = "frame"):
        logger.info(f"Performing temporal search for {len(queries) if queries else 0} queries with group_by_video={group_by_video}")
        
        parsed_queries = []
        if queries:
            for item in queries:
                if isinstance(item, str) and item.strip():
                    parsed_queries.append({"text": item.strip()})
                elif isinstance(item, dict) and item.get("text", "").strip():
                    parsed_queries.append(item)

        if not parsed_queries:
            return self.filter_search(
                ocr_query=ocr_query,
                audio_query=audio_query,
                # audio_match_level=audio_match_level,
                objects=objects,
                group_by_shot=False,
                score_threshold=score_threshold,
                limit=limit
            )
        
        # 1. Search each query with per-event filters (falling back to global filters)
        query_results = []
        for q in parsed_queries:
            q_text = q["text"]
            q_ocr = q.get("ocr") if q.get("ocr") is not None else ocr_query
            q_audio = q.get("audio") if q.get("audio") is not None else audio_query
            q_objects = q.get("objects") if q.get("objects") is not None else objects

            res = self.semantic_search(
                q_text,
                model_names=model_names,
                objects=q_objects,
                top_k=3000,
                score_threshold=score_threshold,
                limit=3000,
                group_by_shot=False,
                ocr_query=q_ocr,
                audio_query=q_audio,
                audio_match_level=audio_match_level
            )
            query_results.append(res)
            
        # 2. Group candidates by video
        video_grouped = {}
        for q_idx, res_list in enumerate(query_results):
            for item in res_list:
                vid = item["video_id"]
                if vid not in video_grouped:
                    video_grouped[vid] = [[] for _ in range(len(parsed_queries))]
                video_grouped[vid][q_idx].append(item)
                
        # 3. Find valid sequence paths via DFS for each video
        all_sequences_by_video = {}
        
        for vid, vid_group in video_grouped.items():
            # If any query has no results for this video, skip
            if any(len(q_res) == 0 for q_res in vid_group):
                continue
            for i in range(len(parsed_queries)):
                vid_group[i].sort(key=lambda x: x["keyframe_index"])
            vid_group_to_search = vid_group

            vid_seqs = []
            def find_paths(current_q_idx, current_path):
                if current_q_idx == len(parsed_queries):
                    vid_seqs.append(list(current_path))
                    return

                for candidate in vid_group_to_search[current_q_idx]:
                    if len(current_path) == 0:
                        current_path.append(candidate)
                        find_paths(current_q_idx + 1, current_path)
                        current_path.pop()
                    else:
                        prev = current_path[-1]
                        gap = candidate["keyframe_index"] - prev["keyframe_index"]
                        if 0 < gap <= MAX_FRAME_GAP:
                            current_path.append(candidate)
                            find_paths(current_q_idx + 1, current_path)
                            current_path.pop()

            find_paths(0, [])
            if vid_seqs:
                all_sequences_by_video[vid] = vid_seqs
            
        # 4. Format output
        final_results = []
        if group_by_video:
            for vid, seq_list in all_sequences_by_video.items():
                scored_seqs = []
                for seq in seq_list:
                    seq_score = sum(item["score"] for item in seq) / len(seq)
                    scored_seqs.append((seq_score, seq))
                
                # Video score = average of all temporal event pair scores in this video
                video_avg_score = sum(s[0] for s in scored_seqs) / len(scored_seqs)
                # Best pair selected for card display
                best_score, best_seq = max(scored_seqs, key=lambda x: x[0])
                
                anchor = best_seq[0]
                final_results.append({
                    "video_id": vid,
                    "sequence_score": video_avg_score,
                    "best_sequence_score": best_score,
                    "frames": best_seq,
                    "display_frames": best_seq,
                    "all_sequences_count": len(seq_list),
                    "keyframe_index": anchor["keyframe_index"],
                    "score": video_avg_score, 
                })
            final_results = sorted(final_results, key=lambda x: x["sequence_score"], reverse=True)
        else:
            for vid, seq_list in all_sequences_by_video.items():
                for seq in seq_list:
                    avg_score = sum(item["score"] for item in seq) / len(seq)
                    anchor = seq[0]
                    final_results.append({
                        "video_id": vid,
                        "sequence_score": avg_score,
                        "best_sequence_score": avg_score,
                        "frames": seq,
                        "display_frames": seq,
                        "all_sequences_count": 1,
                        "keyframe_index": anchor["keyframe_index"],
                        "score": avg_score, 
                    })
            final_results = sorted(final_results, key=lambda x: x["sequence_score"], reverse=True)

        return final_results[:limit]

    def _get_keyframes_in_shot(self, video_id: str, start_frame: int, end_frame: int) -> list:
        """
        List all keyframe indices from data/keyframe/<video_id>/ that fall within [start_frame, end_frame].
        Returns sorted list of keyframe indices.
        """
        kf_dir = self.data_dir / "keyframe" / video_id
        if not kf_dir.exists() or not kf_dir.is_dir():
            return []

        keyframes = []
        for fname in os.listdir(kf_dir):
            if fname.startswith("keyframe_") and fname.endswith(".webp"):
                try:
                    idx = int(fname.replace("keyframe_", "").replace(".webp", ""))
                    if start_frame <= idx <= end_frame:
                        keyframes.append(idx)
                except ValueError:
                    continue
        keyframes.sort()
        return keyframes

    def caption_search(self, query: str, top_k: int = 500, score_threshold: float = 0.0,
                       limit: int = 100, ocr_query=None, audio_query=None, objects: list = None,
                       dense_weight: float = None, bm25_weight: float = None):
        """
        Hybrid semantic & lexical search over shot captions.
        1. Encodes query with Qwen3_Embedding, searches each caption aspect vector in Qdrant.
        2. Normalizes all aspect scores globally (single min-max across all captions of all aspects).
           The shot's dense score is the max normalized score across its aspects, tracking best_aspect.
        3. Runs BM25 text search in Elasticsearch across all aspect fields and caption text with equal weights.
        4. Fuses normalized dense score and normalized BM25 score.
        Returns shot-level results with representative keyframe and best-matching aspect caption.
        """
        logger.info(f"Performing hybrid caption search for: '{query}'")

        if not query or not query.strip():
            return []

        # Encode query using the caption embedding model (Qwen3_Embedding, 1024-dim)
        caption_encoder_name = "Qwen3_Embedding"
        if caption_encoder_name not in self.text_encoders:
            logger.error(f"Caption encoder '{caption_encoder_name}' not initialized. Cannot perform caption search.")
            return []

        encoded = self.text_encoders[caption_encoder_name](query)
        query_vector = encoded[0].tolist()

        # Build OCR/transcript filter ranges
        ocr_ranges = None
        if ocr_query:
            ocr_ranges = self._get_ocr_filter_ranges(ocr_query)
            if not ocr_ranges:
                return []

        transcript_ranges = None
        if audio_query:
            transcript_ranges = self._get_transcript_filter_ranges(audio_query)

        # ─── 1. Dense Aspect Vector Search (Qdrant) ──────────────────────────
        point_aspect_raw = {}
        all_payloads = {}
        all_raw_aspect_scores = []

        for aspect in CAPTION_ASPECT_KEYS:
            try:
                search_result = self.qdrant_client.query_points(
                    collection_name=QDRANT_SHOT_CAPTION_COLLECTION_NAME,
                    query=query_vector,
                    using=aspect,
                    limit=top_k,
                    search_params=SearchParams(exact=True),
                    with_payload=True
                ).points
            except Exception as e:
                logger.warning(f"Caption search failed for aspect '{aspect}': {e}")
                continue

            if not search_result:
                continue

            for hit in search_result:
                all_payloads[hit.id] = hit.payload
                if hit.id not in point_aspect_raw:
                    point_aspect_raw[hit.id] = {}
                point_aspect_raw[hit.id][aspect] = hit.score
                all_raw_aspect_scores.append(hit.score)

        # Global Min-Max Normalization across all captions of all aspects
        dense_shot_scores = {}
        best_aspects = {}
        if all_raw_aspect_scores:
            g_min = min(all_raw_aspect_scores)
            g_max = max(all_raw_aspect_scores)
            g_range = g_max - g_min if g_max > g_min else 1.0

            for point_id, asp_dict in point_aspect_raw.items():
                best_asp = max(asp_dict, key=asp_dict.get)
                max_raw = asp_dict[best_asp]
                norm_score = (max_raw - g_min) / g_range if g_max > g_min else (1.0 if g_max > 0 else 0.0)
                dense_shot_scores[point_id] = norm_score
                best_aspects[point_id] = best_asp

        # ─── 2. Sparse Lexical Search (Elasticsearch BM25) ────────────────────
        bm25_shot_scores = {}
        bm25_data = {}
        try:
            bm25_hits = search_caption_bm25(self.es_client, ES_CAPTION_INDEX_NAME, query, size=top_k)
        except Exception as e:
            logger.warning(f"BM25 caption search failed: {e}")
            bm25_hits = []

        if bm25_hits:
            b_scores = [h["bm25_score"] for h in bm25_hits]
            b_min = min(b_scores)
            b_max = max(b_scores)
            b_range = b_max - b_min if b_max > b_min else 1.0

            for h in bm25_hits:
                vid = h["video_id"]
                s_idx = h["shot_idx"]
                pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{vid}_{s_idx}"))
                norm_bm25 = (h["bm25_score"] - b_min) / b_range if b_max > b_min else 1.0
                bm25_shot_scores[pid] = norm_bm25
                bm25_data[pid] = h

        # ─── 3. Hybrid Fusion ────────────────────────────────────────────────
        w_dense = dense_weight if dense_weight is not None else CAPTION_DENSE_WEIGHT
        w_bm25 = bm25_weight if bm25_weight is not None else CAPTION_BM25_WEIGHT

        if dense_shot_scores and bm25_shot_scores:
            total_w = w_dense + w_bm25
            if total_w > 0:
                w_dense = w_dense / total_w
                w_bm25 = w_bm25 / total_w
        elif dense_shot_scores:
            w_dense = 1.0
            w_bm25 = 0.0
        elif bm25_shot_scores:
            w_dense = 0.0
            w_bm25 = 1.0
        else:
            return []

        all_pids = set(dense_shot_scores.keys()) | set(bm25_shot_scores.keys())
        final_scores = {}
        for pid in all_pids:
            s_dense = dense_shot_scores.get(pid, 0.0)
            s_bm25 = bm25_shot_scores.get(pid, 0.0)
            final_scores[pid] = w_dense * s_dense + w_bm25 * s_bm25

        # ─── 4. Payload resolution & Filtering ───────────────────────────────
        for pid in all_pids:
            if pid not in all_payloads and pid in bm25_data:
                h = bm25_data[pid]
                all_payloads[pid] = {
                    "video_id": h["video_id"],
                    "shot_idx": h["shot_idx"],
                    "start_frame": h.get("start_frame", 0),
                    "end_frame": h.get("end_frame", 0),
                    "representative_frame": h.get("representative_frame", 0),
                    "keyframe_idx": h.get("representative_frame", 0),
                    "start_sec": h.get("start_sec"),
                    "end_sec": h.get("end_sec"),
                    "caption_text": h.get("caption_text", ""),
                    "aspects": h.get("aspects", {}),
                    "ocr_detected": h.get("ocr_detected", []),
                    "audio_transcript": h.get("audio_transcript", ""),
                }

        # Apply OCR filter
        if ocr_ranges is not None:
            filtered = {}
            for pid, score in final_scores.items():
                payload = all_payloads.get(pid, {})
                vid = payload.get("video_id")
                rep_frame = payload.get("representative_frame", payload.get("keyframe_idx", 0))
                if self._frame_in_ocr_ranges(vid, rep_frame, ocr_ranges) > 0.0:
                    filtered[pid] = score
            final_scores = filtered

        # Apply transcript filter
        if transcript_ranges is not None:
            filtered = {}
            for pid, score in final_scores.items():
                payload = all_payloads.get(pid, {})
                vid = payload.get("video_id")
                rep_frame = payload.get("representative_frame", payload.get("keyframe_idx", 0))
                if self._frame_in_transcript_ranges(vid, rep_frame, transcript_ranges) > 0.0:
                    filtered[pid] = score
            final_scores = filtered

        # ─── 5. Rank and format results ──────────────────────────────────────
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for point_id, score in ranked:
            if score < score_threshold:
                continue

            payload = all_payloads.get(point_id, {})
            video_id = payload.get("video_id")
            shot_idx = payload.get("shot_idx", 0)
            start_frame = payload.get("start_frame", 0)
            end_frame = payload.get("end_frame", 0)
            rep_frame = payload.get("representative_frame", payload.get("keyframe_idx", 0))

            # Get the best aspect's caption text
            best_aspect_key = best_aspects.get(point_id, "tong_the_canh_quay")
            aspects_data = payload.get("aspects", {})
            best_caption = aspects_data.get(best_aspect_key, "")
            if not best_caption:
                best_caption = payload.get("caption_text", "")

            results.append({
                "video_id": video_id,
                "keyframe_index": rep_frame,
                "score": score,
                "dense_score": dense_shot_scores.get(point_id, 0.0),
                "bm25_score": bm25_shot_scores.get(point_id, 0.0),
                "shot_idx": shot_idx,
                "shot_start_frame": start_frame,
                "shot_end_frame": end_frame,
                "representative_frame": rep_frame,
                "caption_text": payload.get("caption_text", ""),
                "best_aspect": best_aspect_key,
                "best_aspect_caption": best_caption,
                "aspects": aspects_data,
                "result_type": "caption",
            })

            if len(results) >= limit:
                break

        return results


    def fused_search(self, query: str, model_names: list = None, objects: list = None,
                     top_k: int = 1000, score_threshold: float = 0.0,
                     limit: int = 100, ocr_query=None, audio_query=None,
                     keyframe_weight: float = None, caption_weight: float = None):
        """
        Fused search: combine keyframe embedding search and caption embedding search.
        For each keyframe in the union of both result sets, compute:
          fused_score = w_kf * normalized_keyframe_score + w_cap * normalized_caption_score
        Caption search expands shot results to ALL keyframes in that shot's range.
        """
        from config import KEYFRAME_SEARCH_WEIGHT, CAPTION_SEARCH_WEIGHT

        w_kf = keyframe_weight if keyframe_weight is not None else KEYFRAME_SEARCH_WEIGHT
        w_cap = caption_weight if caption_weight is not None else CAPTION_SEARCH_WEIGHT

        logger.info(f"Performing fused search for: '{query}' (kf_weight={w_kf}, cap_weight={w_cap})")

        # 1. Run keyframe search (returns per-keyframe results)
        kf_results = self.semantic_search(
            query, model_names=model_names, objects=objects,
            top_k=top_k, score_threshold=0.0,
            limit=top_k, group_by_shot=False,
            ocr_query=ocr_query, audio_query=audio_query
        )

        # 2. Run caption search (returns per-shot results)
        cap_results = self.caption_search(
            query, top_k=top_k, score_threshold=0.0,
            limit=top_k, ocr_query=ocr_query, audio_query=audio_query,
            objects=objects
        )

        # 3. Build keyframe score maps

        # Keyframe scores: (video_id, keyframe_idx) -> raw score
        kf_score_map = {}
        kf_extra = {}  # store extra info like ocr_text, audio_text
        for r in kf_results:
            key = (r["video_id"], r["keyframe_index"])
            kf_score_map[key] = r["score"]
            kf_extra[key] = {
                "ocr_text": r.get("ocr_text", ""),
                "audio_text": r.get("audio_text", ""),
            }

        # Caption scores: expand each shot to all keyframes in range
        # (video_id, keyframe_idx) -> caption score (same for all kf in shot)
        cap_score_map = {}
        cap_shot_info = {}  # (video_id, keyframe_idx) -> shot metadata
        for r in cap_results:
            vid = r["video_id"]
            start_f = r["shot_start_frame"]
            end_f = r["shot_end_frame"]
            shot_score = r["score"]

            # Get ALL keyframes in this shot's range from disk
            shot_keyframes = self._get_keyframes_in_shot(vid, start_f, end_f)
            if not shot_keyframes:
                # Fallback to representative frame
                shot_keyframes = [r["representative_frame"]]

            for kf_idx in shot_keyframes:
                key = (vid, kf_idx)
                # If keyframe appears in multiple shots (shouldn't happen), take max
                if key not in cap_score_map or shot_score > cap_score_map[key]:
                    cap_score_map[key] = shot_score
                    cap_shot_info[key] = {
                        "shot_idx": r.get("shot_idx", 0),
                        "shot_start_frame": start_f,
                        "shot_end_frame": end_f,
                        "caption_text": r.get("caption_text", ""),
                        "best_aspect": r.get("best_aspect", ""),
                        "best_aspect_caption": r.get("best_aspect_caption", ""),
                    }

        # 4. Union of all keyframe keys
        all_keys = set(kf_score_map.keys()) | set(cap_score_map.keys())

        if not all_keys:
            return []

        # 5. Min-Max normalize each score map independently
        def _minmax_normalize(score_map):
            if not score_map:
                return {}
            values = list(score_map.values())
            min_v = min(values)
            max_v = max(values)
            if max_v == min_v:
                return {k: (1.0 if max_v > 0 else 0.0) for k, v in score_map.items()}
            return {k: (v - min_v) / (max_v - min_v) for k, v in score_map.items()}

        kf_norm = _minmax_normalize(kf_score_map)
        cap_norm = _minmax_normalize(cap_score_map)

        # 6. Compute fused score for each keyframe
        fused_results = []
        for key in all_keys:
            vid, kf_idx = key
            kf_s = kf_norm.get(key, 0.0)
            cap_s = cap_norm.get(key, 0.0)
            fused = w_kf * kf_s + w_cap * cap_s

            if fused < score_threshold:
                continue

            # Determine shot boundaries
            shot_start = 0
            shot_end = 0
            if key in cap_shot_info:
                shot_start = cap_shot_info[key]["shot_start_frame"]
                shot_end = cap_shot_info[key]["shot_end_frame"]
            elif vid in self.shots_data:
                boundaries = self.shots_data[vid]
                starts = [b[0] for b in boundaries]
                idx = bisect.bisect_right(starts, kf_idx) - 1
                if idx >= 0 and kf_idx <= boundaries[idx][1]:
                    shot_start = boundaries[idx][0]
                    shot_end = boundaries[idx][1]

            extra = kf_extra.get(key, {})
            cap_info = cap_shot_info.get(key, {})

            fused_results.append({
                "video_id": vid,
                "keyframe_index": kf_idx,
                "score": fused,
                "keyframe_score": kf_s,
                "caption_score": cap_s,
                "shot_start_frame": shot_start,
                "shot_end_frame": shot_end,
                "ocr_text": extra.get("ocr_text", ""),
                "audio_text": extra.get("audio_text", ""),
                "caption_text": cap_info.get("caption_text", ""),
                "best_aspect": cap_info.get("best_aspect", ""),
                "best_aspect_caption": cap_info.get("best_aspect_caption", ""),
                "result_type": "fused",
            })

        fused_results.sort(key=lambda x: x["score"], reverse=True)
        return fused_results[:limit]

if __name__ == "__main__":
    system = RetrievalSystem(init_text_encoders=False)
    # results = system.semantic_search("A person riding a horse on a beach.")
    # system.ingest_shot_caption_embedding()
    system.ingest_all_vision_embedding(max_workers=12)