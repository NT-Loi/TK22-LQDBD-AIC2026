import logging
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Configure logging to output to both console and a file called 'app.log'
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
for _handler in logging.getLogger().handlers:
    _stream = getattr(_handler, "stream", None)
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import os
import torch
import uuid
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchAny, Range, SetPayloadOperation, SetPayload, SearchParams, QueryRequest, DenseVectorConfig, DenseVectorNameConfig, Distance
from elasticsearch import Elasticsearch

from data_processor.text_encoder import *

from config import *
from utils import *
from utils.video_metadata import load_video_metadata
import json
import bisect
import collections
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ThreadSafeLRUCache:
    """Thread-safe high-capacity LRU Cache for query embeddings."""
    def __init__(self, maxsize: int = 8192):
        self.maxsize = maxsize
        self.cache = collections.OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

    def __len__(self):
        with self.lock:
            return len(self.cache)

class RetrievalSystem:
    def __init__(self, data_dir: str=DATA_DIR, device=None, re_ingest: bool=False, init_text_encoders: bool=True):
        logger.info("Initializing Video Retrieval System...")

        # auto use gpu
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.data_dir = Path(data_dir)
        self.keyframe_dir = self.data_dir / "keyframe"
        self.vision_embedding_dir = self.data_dir / "embedding"
        self.caption_dir = self.data_dir / "caption"
        self.caption_embedding_dir = self.data_dir / "caption_embedding"
        self.shot_dir = self.data_dir / "shot"
        self.obj_dir = self.data_dir / "object_detection"
        self.ocr_dir = self.data_dir / "ocr"
        self.transcript_dir = self.data_dir / "transcript"

        self.qdrant_client = QdrantClient(url=QDRANT_HOST_URL, prefer_grpc=True, timeout=60.0)
        
        self.vision_models = VISION_MODELS
        if not self.qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
            logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' does not exist. Please run ingestion.")
        else:
            try:
                coll_info = self.qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
                existing_vectors = coll_info.config.params.vectors or {}
                for m_name, m_cfg in VISION_MODELS.items():
                    if m_name not in existing_vectors:
                        logger.info(f"Adding missing named vector '{m_name}' (dim={m_cfg['dim']}) to Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
                        self.qdrant_client.create_vector_name(
                            collection_name=QDRANT_COLLECTION_NAME,
                            vector_name=m_name,
                            vector_name_config=DenseVectorNameConfig(
                                dense=DenseVectorConfig(size=m_cfg["dim"], distance=Distance.COSINE)
                            ),
                            wait=True,
                        )
            except Exception as e:
                logger.warning(f"Could not verify/create collection vector names: {e}")

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
                try:
                    if model_name == "CLIP_H14":
                        self.text_encoders[model_name] = CLIP_H14TextEncoder(device=self.device)
                    elif model_name in ("CLIP", "CLIP_B32"):
                        self.text_encoders[model_name] = CLIPTextEncoder(device=self.device)
                    elif model_name == "SigLIP":
                        self.text_encoders[model_name] = SigLIPTextEncoder(device=self.device)
                    elif model_name == "SigLIP2":
                        self.text_encoders[model_name] = SigLIP2TextEncoder(device=self.device)
                    elif model_name == "Qwen3_VL_Embedding":
                        self.text_encoders[model_name] = Qwen3VLEmbeddingTextEncoder(device=self.device)
                    elif model_name == "FG_CLIP2":
                        self.text_encoders[model_name] = FGCLIP2TextEncoder(device=self.device)
                    elif model_name == "Qwen3_Embedding":
                        self.text_encoders[model_name] = Qwen3EmbeddingTextEncoder(device=self.device)
                except Exception:
                    logger.exception("Failed to initialize text encoder '%s'; continuing without it.", model_name)

        # High-capacity thread-safe LRU cache for query text embeddings (up to 8,192 queries)
        self.query_cache = ThreadSafeLRUCache(maxsize=8192)

        # Pre-load shot boundaries on RAM to efficiently group frame by
        self.shots_data = {}
        self.shot_starts = {}
        self._load_shots()

        # Pre-index sorted keyframe indices per video into RAM for fast O(log K) shot expansion
        self.video_keyframes = {}
        self._load_video_keyframes()

    def _load_video_keyframes(self):
        """Pre-index keyframes, reusing cached entries for unchanged video directories."""
        kf_base = getattr(self, "keyframe_dir", self.data_dir / "keyframe")
        if not kf_base.exists() or not kf_base.is_dir():
            logger.warning(f"Keyframe directory {kf_base} does not exist.")
            return

        logger.info("Pre-indexing video keyframes into RAM...")
        cache_path = self.data_dir / ".cache" / "keyframe_index.json"
        cached_videos = {}
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached_data.get("version") == 1:
                cached_videos = cached_data.get("videos", {})
        except (OSError, ValueError, TypeError):
            pass

        count = 0
        cache_out = {}
        try:
            for vid_entry in os.scandir(kf_base):
                if vid_entry.is_dir():
                    vid = vid_entry.name
                    mtime_ns = vid_entry.stat().st_mtime_ns
                    cached = cached_videos.get(vid, {})
                    if cached.get("mtime_ns") == mtime_ns:
                        indices = cached.get("indices", [])
                    else:
                        indices = []
                        for f_entry in os.scandir(vid_entry.path):
                            fname = f_entry.name
                            if fname.startswith("keyframe_") and fname.endswith(".webp"):
                                try:
                                    indices.append(int(fname[9:-5]))
                                except ValueError:
                                    continue
                        indices.sort()
                    self.video_keyframes[vid] = indices
                    cache_out[vid] = {"mtime_ns": mtime_ns, "indices": indices}
                    count += 1
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_suffix(".tmp")
                tmp_path.write_text(
                    json.dumps({"version": 1, "videos": cache_out}, separators=(",", ":")),
                    encoding="utf-8",
                )
                os.replace(tmp_path, cache_path)
            except OSError as e:
                logger.warning("Could not persist keyframe index cache: %s", e)
            logger.info(f"Pre-indexed keyframes for {count} videos into RAM.")
        except Exception as e:
            logger.error(f"Error pre-indexing keyframes: {e}")

    def _cached_encode(self, model_name: str, query: str):
        """Encode query using specified model with thread-safe high-capacity LRU caching."""
        if not hasattr(self, "query_cache"):
            self.query_cache = ThreadSafeLRUCache(maxsize=8192)

        clean_q = query.strip()
        cache_key = (model_name, clean_q)
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        encoded = self.text_encoders[model_name](clean_q)
        self.query_cache.set(cache_key, encoded)
        return encoded.copy()

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
                            sorted_boundaries = sorted(boundaries, key=lambda x: x[0])
                            self.shots_data[video_id] = sorted_boundaries
                            self.shot_starts[video_id] = [boundary[0] for boundary in sorted_boundaries]
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")

    def process_video_data(self):
        pass

    def ingest_vision_embedding(self, embedding_model: str, max_workers: int = 8):
        model_dirs = [Path(d) / embedding_model for d in EMBEDDING_DIRS if (Path(d) / embedding_model).is_dir()]
        if not model_dirs:
            logger.warning(f"Embedding directory for {embedding_model} not found in any EMBEDDING_DIRS. Skipping.")
            return

        expected_dim = VISION_EMBEDDING_DIM.get(embedding_model)
        video_dirs_map = {}
        for md in model_dirs:
            for d in md.iterdir():
                if d.is_dir() and d.name not in video_dirs_map:
                    video_dirs_map[d.name] = d
        video_dirs = sorted(video_dirs_map.values(), key=lambda d: d.name)

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
        Encode query using specified text encoders with LRU cache and multi-thread parallelization.
        If model_names is None, it uses all initialized text encoders.
        Returns a dictionary mapping model_name to encoded text vector.
        """
        if model_names is None:
            model_names = list(self.text_encoders.keys())
            
        encoded_vectors = {}
        valid_models = [m for m in model_names if m in self.text_encoders]

        if len(valid_models) > 1:
            def _enc(m):
                return m, self._cached_encode(m, query)
            with ThreadPoolExecutor(max_workers=min(len(valid_models), 4)) as executor:
                futures = [executor.submit(_enc, m) for m in valid_models]
                for fut in as_completed(futures):
                    m, vec = fut.result()
                    encoded_vectors[m] = vec
        else:
            for model_name in valid_models:
                encoded_vectors[model_name] = self._cached_encode(model_name, query)

        for model_name in model_names:
            if model_name not in self.text_encoders:
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

    def _is_frame_in_same_shot(self, video_id: str, kf1: int, kf2: int) -> bool:
        """
        Check if two keyframes belong to the same shot in video_id using shot boundary data.
        """
        if not hasattr(self, "shots_data") or video_id not in self.shots_data:
            return False
        boundaries = self.shots_data[video_id]
        if not boundaries:
            return False
        starts = getattr(self, "shot_starts", {}).get(video_id)
        if starts is None:
            starts = [b[0] for b in boundaries]
        idx1 = bisect.bisect_right(starts, kf1) - 1
        if idx1 >= 0 and kf1 <= boundaries[idx1][1]:
            return boundaries[idx1][0] <= kf2 <= boundaries[idx1][1]
        return False

    def _get_text_filter_ranges(self, text_filters, model_names: list = None, top_k: int = 3000) -> dict:
        """
        Get text filter data for single or multiple visual text queries with per-query level ('frame' or 'video').
        Commutative AND logic: order of queries is non-temporal and has no bearing on chronological sequence.
        Returns dict with 'common_vids' and 'query_items'.
        """
        if isinstance(text_filters, dict) and "common_vids" in text_filters:
            return text_filters

        parsed_queries = []
        if isinstance(text_filters, str):
            if text_filters.strip():
                parsed_queries.append({"text": text_filters.strip(), "level": "frame"})
        elif isinstance(text_filters, list):
            for item in text_filters:
                if isinstance(item, str) and item.strip():
                    parsed_queries.append({"text": item.strip(), "level": "frame"})
                elif isinstance(item, dict) and item.get("text", "").strip():
                    lvl = item.get("level", "frame")
                    if lvl not in ("frame", "video"):
                        lvl = "frame"
                    parsed_queries.append({"text": item["text"].strip(), "level": lvl})

        if not parsed_queries:
            return {}

        # Select primary vision text encoder
        vision_model_names = [m for m in (model_names or []) if m in self.text_encoders and m in VISION_MODELS]
        if not vision_model_names:
            vision_model_names = [m for m in ["SigLIP2", "SigLIP", "Qwen3_VL_Embedding", "CLIP_H14", "CLIP", "FG_CLIP2"] if m in self.text_encoders]
        if not vision_model_names:
            vision_model_names = list(self.text_encoders.keys())
        if not vision_model_names:
            logger.error("No text encoders available for text filter.")
            return {}

        primary_model = vision_model_names[0]
        query_items = []

        def _fetch_filter(q_item):
            try:
                encoded = self._cached_encode(primary_model, q_item["text"])
                query_vector = encoded[0].tolist()

                search_result = self.qdrant_client.query_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query=query_vector,
                    using=primary_model,
                    limit=top_k,
                    search_params=SearchParams(exact=False, hnsw_ef=HNSW_EF_SEARCH),
                    with_payload=True
                ).points
                return q_item, search_result
            except Exception as e:
                logger.error(f"Error executing Qdrant query for text filter '{q_item['text']}': {e}")
                return q_item, []

        query_hits = []
        if len(parsed_queries) > 1:
            with ThreadPoolExecutor(max_workers=min(len(parsed_queries), 4)) as executor:
                futures = [executor.submit(_fetch_filter, q) for q in parsed_queries]
                for fut in as_completed(futures):
                    q_item, search_result = fut.result()
                    if search_result:
                        query_hits.append((q_item, search_result))
        else:
            q_item, search_result = _fetch_filter(parsed_queries[0])
            if search_result:
                query_hits.append((q_item, search_result))

        for q, search_result in query_hits:
            scores = [hit.score for hit in search_result]
            min_score = min(scores)
            max_score = max(scores)

            vid_map = {}
            for hit in search_result:
                if max_score == min_score:
                    norm_score = 1.0 if max_score > 0 else 0.0
                else:
                    norm_score = (hit.score - min_score) / (max_score - min_score)

                # Keep candidates with normalized score >= 0.15 (exclude tail noise)
                if norm_score < 0.15:
                    continue

                vid = hit.payload.get("video_id")
                kf_idx = hit.payload.get("keyframe_idx", hit.payload.get("frame_idx", 0))
                if not vid:
                    continue

                if vid not in vid_map:
                    vid_map[vid] = {}
                vid_map[vid][kf_idx] = {
                    "score": norm_score,
                    "raw_score": hit.score,
                    "text": q["text"]
                }

            query_items.append({
                "text": q["text"],
                "level": q["level"],
                "vid_map": vid_map
            })

        if not query_items:
            return {}

        # AND Logic: Keep only videos present in ALL per_query_text_filter_results
        common_vids = set(query_items[0]["vid_map"].keys())
        for q_item in query_items[1:]:
            common_vids &= set(q_item["vid_map"].keys())

        if not common_vids:
            return {}

        return {
            "common_vids": common_vids,
            "query_items": query_items
        }

    def _frame_in_text_filter_ranges(self, video_id: str, keyframe_idx: int, text_filter_data: dict) -> float:
        """
        Check if a keyframe passes all text filters (AND logic across queries, per-query level).
        Order of text filter queries is non-temporal and has no bearing on chronological sequence.
        Returns combined score (>0.0) if passed, or 0.0 if failed.
        """
        if not text_filter_data or "common_vids" not in text_filter_data:
            return 0.0

        if video_id not in text_filter_data["common_vids"]:
            return 0.0

        total_score = 0.0

        for q_item in text_filter_data["query_items"]:
            level = q_item["level"]
            vid_map = q_item["vid_map"]
            frames_dict = vid_map.get(video_id, {})

            if level == "video":
                # Video level: satisfied if video matched this text query anywhere
                if not frames_dict:
                    return 0.0
                max_sc = max(info["score"] for info in frames_dict.values())
                total_score += max_sc
            else:
                # Frame level: keyframe itself must match this text query (or within proximity window / shot)
                if keyframe_idx in frames_dict:
                    total_score += frames_dict[keyframe_idx]["score"]
                else:
                    matched_sc = 0.0
                    for kf, info in frames_dict.items():
                        if abs(kf - keyframe_idx) <= 150:  # ~5s frame window
                            matched_sc = max(matched_sc, info["score"])
                        elif self._is_frame_in_same_shot(video_id, kf, keyframe_idx):
                            matched_sc = max(matched_sc, info["score"])

                    if matched_sc > 0.0:
                        total_score += matched_sc
                    else:
                        return 0.0

        return total_score

    def filter_search(self, ocr_query: str = None, audio_query = None, objects: list = None,
                      text_filters = None, model_names: list = None,
                      group_by_shot: bool = False, score_threshold: float = 0.0, limit: int = 100):
        """
        Perform search purely using metadata filters (OCR, audio transcript, text query filter, object detection)
        without requiring a text query.
        """
        logger.info(f"Performing filter search with ocr_query='{ocr_query}', audio_query='{audio_query}', text_filters='{text_filters}', objects={objects}")

        if not ocr_query and not audio_query and not objects and not text_filters:
            return []

        # 1. Gather transcript ranges if audio_query is given
        transcript_ranges = None
        if audio_query:
            transcript_ranges = self._get_transcript_filter_ranges(audio_query)
            if not transcript_ranges and not ocr_query and not objects and not text_filters:
                return []

        # 2. Gather OCR ranges if ocr_query is given
        ocr_ranges = None
        if ocr_query:
            ocr_ranges = self._get_ocr_filter_ranges(ocr_query)
            if not ocr_ranges and not audio_query and not objects and not text_filters:
                return []

        # 3. Gather Text Filter ranges if text_filters is given
        text_filter_ranges = None
        if text_filters:
            text_filter_ranges = self._get_text_filter_ranges(text_filters, model_names=model_names)
            if not text_filter_ranges and not ocr_query and not audio_query and not objects:
                return []

        # Map: (video_id, keyframe_idx) -> score
        candidates = {}

        if text_filter_ranges:
            # Case A: Text query filter provided
            for vid in text_filter_ranges["common_vids"]:
                if ocr_ranges and vid not in ocr_ranges["common_vids"]:
                    continue
                if transcript_ranges and vid not in transcript_ranges["common_vids"]:
                    continue

                # Collect candidate frames for this video
                frame_candidates = set()
                for q_item in text_filter_ranges["query_items"]:
                    if q_item["level"] == "frame":
                        frame_candidates.update(q_item["vid_map"].get(vid, {}).keys())
                if not frame_candidates:
                    for q_item in text_filter_ranges["query_items"]:
                        frame_candidates.update(q_item["vid_map"].get(vid, {}).keys())

                for kf_idx in frame_candidates:
                    txt_sc = self._frame_in_text_filter_ranges(vid, kf_idx, text_filter_ranges)
                    if txt_sc <= 0.0:
                        continue

                    ocr_sc = 0.0
                    if ocr_ranges:
                        ocr_sc = self._frame_in_ocr_ranges(vid, kf_idx, ocr_ranges)
                        if ocr_sc <= 0.0:
                            continue

                    aud_sc = 0.0
                    if transcript_ranges:
                        aud_sc = self._frame_in_transcript_ranges(vid, kf_idx, transcript_ranges)
                        if aud_sc <= 0.0:
                            continue

                    candidates[(vid, kf_idx)] = txt_sc + ocr_sc + aud_sc

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

        elif ocr_ranges:
            # Case B: OCR query provided
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
            # Case C: Audio query provided (no OCR query, no text filters)
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
            # Case D: Only object filter provided (no OCR, no audio, no text filters)
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

    def semantic_search(self, query: str, model_names: list = None, objects: list = None, top_k: int = 1000, score_threshold: float = 0.0, group_by_shot: bool = False, limit: int = 100, ocr_query: str = None, audio_query = None, audio_match_level: str = "frame", text_filters = None):
        logger.info(f"Performing semantic search for: '{query}' with model(s): {model_names}, ocr_query: '{ocr_query}', audio_query: '{audio_query}', match_level: '{audio_match_level}', text_filters: '{text_filters}'.")
        
        if not query or not query.strip():
            return self.filter_search(
                ocr_query=ocr_query,
                audio_query=audio_query,
                # audio_match_level=audio_match_level,
                objects=objects,
                text_filters=text_filters,
                model_names=model_names,
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

        # Build text filter ranges if text_filters is provided
        text_filter_ranges = None
        if text_filters:
            text_filter_ranges = self._get_text_filter_ranges(text_filters, model_names=model_names)
            if not text_filter_ranges:
                logger.info("No matches for text_filters. Returning empty results.")
                return []
        
        # Only encode with vision-compatible models (those with entries in VISION_MODELS)
        if model_names is None or not model_names:
            vision_model_names = [DEFAULT_VISION_MODEL] if DEFAULT_VISION_MODEL in self.text_encoders else [m for m in self.text_encoders.keys() if m in VISION_MODELS]
        elif "all" in model_names:
            vision_model_names = [m for m in self.text_encoders.keys() if m in VISION_MODELS]
        else:
            vision_model_names = [m for m in model_names if m in VISION_MODELS]
            if not vision_model_names and DEFAULT_VISION_MODEL in self.text_encoders:
                vision_model_names = [DEFAULT_VISION_MODEL]

        encoded_vectors = self.encode_query(query, vision_model_names)

        # Parallel Qdrant queries across models using HNSW approximate search
        def _query_model(m_name, vector):
            q_vec = vector[0].tolist()
            payload_fields = ["video_id", "keyframe_idx", "frame_idx"]
            if objects:
                payload_fields.extend(["objects", "class_counts"])
            res = self.qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=q_vec,
                using=m_name,
                limit=top_k,
                search_params=SearchParams(exact=False, hnsw_ef=HNSW_EF_SEARCH),
                with_payload=payload_fields
            ).points
            if objects:
                res = [hit for hit in res if self._matches_object_filters(hit.payload, objects)]
            return m_name, res

        model_results = {}
        if len(encoded_vectors) > 1:
            # Send all vector searches in one Qdrant request. This preserves
            # each independent nearest-neighbour result while avoiding one
            # transport round trip per model.
            try:
                is_real_qdrant_client = isinstance(self.qdrant_client, QdrantClient)
                if not is_real_qdrant_client:
                    raise TypeError("batch API unavailable on the injected Qdrant client")
                model_items = list(encoded_vectors.items())
                payload_fields = ["video_id", "keyframe_idx", "frame_idx"]
                if objects:
                    payload_fields.extend(["objects", "class_counts"])
                batch_responses = self.qdrant_client.query_batch_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    requests=[
                        QueryRequest(
                            query=vector[0].tolist(),
                            using=model_name,
                            limit=top_k,
                            params=SearchParams(exact=False, hnsw_ef=HNSW_EF_SEARCH),
                            with_payload=payload_fields,
                        )
                        for model_name, vector in model_items
                    ],
                )
                for (model_name, _), response in zip(model_items, batch_responses):
                    res = response.points
                    if objects:
                        res = [hit for hit in res if self._matches_object_filters(hit.payload, objects)]
                    if res:
                        model_results[model_name] = res
            except Exception as exc:
                if isinstance(self.qdrant_client, QdrantClient):
                    logger.warning("Batched Qdrant query failed; retrying per model: %s", exc)
                with ThreadPoolExecutor(max_workers=len(encoded_vectors)) as executor:
                    futures = [executor.submit(_query_model, m, v) for m, v in encoded_vectors.items()]
                    for future in futures:
                        model_name, res = future.result()
                        if res:
                            model_results[model_name] = res
        elif len(encoded_vectors) == 1:
            m, v = next(iter(encoded_vectors.items()))
            _, res = _query_model(m, v)
            if res:
                model_results[m] = res

        normalized_results = {}
        all_payloads = {}

        for model_name, search_result in model_results.items():
            if model_name not in EMBEDDING_WEIGHTS:
                logger.warning(f"Embedding weight for {model_name} hasn't been initialized. Use 1.0 as default value.")
                EMBEDDING_WEIGHTS[model_name] = 1.0

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
            
        # Calculate overall score: normalize by the sum of weights of models that actually returned this keyframe
        point_weighted_sum = {}
        point_weight_sum = {}
        for model_name, scores_dict in normalized_results.items():
            weight = EMBEDDING_WEIGHTS.get(model_name, 1.0)
            for point_id, norm_score in scores_dict.items():
                if point_id not in point_weighted_sum:
                    point_weighted_sum[point_id] = 0.0
                    point_weight_sum[point_id] = 0.0
                point_weighted_sum[point_id] += float(norm_score * weight)
                point_weight_sum[point_id] += float(weight)

        final_scores = {}
        for point_id, w_sum in point_weight_sum.items():
            if w_sum > 0:
                final_scores[point_id] = point_weighted_sum[point_id] / w_sum
            else:
                final_scores[point_id] = 0.0
                
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

        # Apply Text filter intersection filter before ranking
        if text_filter_ranges is not None:
            final_scores = {
                pid: score for pid, score in final_scores.items()
                if self._frame_in_text_filter_ranges(
                    all_payloads[pid].get("video_id"),
                    all_payloads[pid].get("keyframe_idx", all_payloads[pid].get("frame_idx", 0)),
                    text_filter_ranges
                ) > 0.0
            }
            logger.info(f"After text filter: {len(final_scores)} results remain.")

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
                starts = getattr(self, "shot_starts", {}).get(video_id)
                if starts is None:
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
            if not group_by_shot and len(results) >= limit:
                break
            
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

    def temporal_search(self, queries: list, model_names: list = None, objects: list = None, group_by_video: bool = False, score_threshold: float = 0.0, limit: int = 100, ocr_query = None, audio_query = None, audio_match_level: str = "frame", text_filters = None):
        logger.info(f"Performing temporal search for {len(queries) if queries else 0} queries with group_by_video={group_by_video}, text_filters={text_filters}")
        
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
                text_filters=text_filters,
                model_names=model_names,
                group_by_shot=False,
                score_threshold=score_threshold,
                limit=limit
            )

        # Pre-compute text_filter_ranges once for all event sub-queries
        text_filter_ranges = None
        if text_filters:
            text_filter_ranges = self._get_text_filter_ranges(text_filters, model_names=model_names)
            if not text_filter_ranges:
                logger.info("No matches for text_filters in temporal_search. Returning empty results.")
                return []
        
        # 1. Search each query with per-event filters concurrently
        def _search_single_query(q):
            q_text = q["text"]
            q_ocr = q.get("ocr") if q.get("ocr") is not None else ocr_query
            q_audio = q.get("audio") if q.get("audio") is not None else audio_query
            q_objects = q.get("objects") if q.get("objects") is not None else objects

            return self.semantic_search(
                q_text,
                model_names=model_names,
                objects=q_objects,
                top_k=3000,
                score_threshold=score_threshold,
                limit=3000,
                group_by_shot=False,
                ocr_query=q_ocr,
                audio_query=q_audio,
                audio_match_level=audio_match_level,
                text_filters=text_filter_ranges
            )

        if len(parsed_queries) > 1:
            with ThreadPoolExecutor(max_workers=len(parsed_queries)) as executor:
                query_results = list(executor.map(_search_single_query, parsed_queries))
        else:
            query_results = [_search_single_query(parsed_queries[0])]
            
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
        Return sorted list of keyframe indices for video_id that fall within [start_frame, end_frame].
        Uses in-memory pre-indexed keyframes with O(log K) binary search for maximum speed.
        """
        if hasattr(self, "video_keyframes") and self.video_keyframes:
            kfs = self.video_keyframes.get(video_id)
            if kfs:
                i = bisect.bisect_left(kfs, start_frame)
                j = bisect.bisect_right(kfs, end_frame)
                return kfs[i:j]
            return []

        # Fallback to disk if video_keyframes is not populated
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
                       dense_weight: float = None, bm25_weight: float = None, text_filters = None):
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

        # Auto-translate English query to Vietnamese if needed (Qwen3 & BM25 alignment)
        from config import AUTO_TRANSLATE_EN_CAPTION
        from utils.translator import translate_query_to_vi_if_needed

        search_query = query
        translated_query_str = None
        if AUTO_TRANSLATE_EN_CAPTION:
            trans_text, was_en = translate_query_to_vi_if_needed(query)
            if was_en:
                logger.info("Caption search auto-translated EN query: '%s' -> '%s'", query, trans_text)
                search_query = trans_text
                translated_query_str = trans_text

        # Encode query using the caption embedding model (Qwen3_Embedding, 1024-dim) with LRU caching
        caption_encoder_name = "Qwen3_Embedding"
        if caption_encoder_name not in self.text_encoders:
            logger.error(f"Caption encoder '{caption_encoder_name}' not initialized. Cannot perform caption search.")
            return []

        query_vector = self._cached_encode(caption_encoder_name, search_query)[0].tolist()

        # Build OCR/transcript filter ranges
        ocr_ranges = None
        if ocr_query:
            ocr_ranges = self._get_ocr_filter_ranges(ocr_query)
            if not ocr_ranges:
                return []

        transcript_ranges = None
        if audio_query:
            transcript_ranges = self._get_transcript_filter_ranges(audio_query)

        text_filter_ranges = None
        if text_filters:
            text_filter_ranges = self._get_text_filter_ranges(text_filters)
            if not text_filter_ranges:
                return []

        # ─── 1. Concurrent Dense Aspect Vector Search (Qdrant) & Lexical (BM25) ───
        point_aspect_raw = {}
        all_payloads = {}
        all_raw_aspect_scores = []

        def _search_aspect(asp):
            try:
                res = self.qdrant_client.query_points(
                    collection_name=QDRANT_SHOT_CAPTION_COLLECTION_NAME,
                    query=query_vector,
                    using=asp,
                    limit=top_k,
                    search_params=SearchParams(exact=False, hnsw_ef=HNSW_EF_SEARCH),
                    with_payload=True
                ).points
                return asp, res
            except Exception as e:
                logger.warning(f"Caption search failed for aspect '{asp}': {e}")
                return asp, []

        def _search_bm25():
            try:
                return search_caption_bm25(self.es_client, ES_CAPTION_INDEX_NAME, search_query, size=top_k)
            except Exception as e:
                logger.warning(f"BM25 caption search failed: {e}")
                return []

        with ThreadPoolExecutor(max_workers=len(CAPTION_ASPECT_KEYS) + 1) as executor:
            aspect_futures = [executor.submit(_search_aspect, asp) for asp in CAPTION_ASPECT_KEYS]
            bm25_future = executor.submit(_search_bm25)

            for f in aspect_futures:
                aspect, search_result = f.result()
                if not search_result:
                    continue
                for hit in search_result:
                    all_payloads[hit.id] = hit.payload
                    if hit.id not in point_aspect_raw:
                        point_aspect_raw[hit.id] = {}
                    point_aspect_raw[hit.id][aspect] = hit.score
                    all_raw_aspect_scores.append(hit.score)

            bm25_hits = bm25_future.result()

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

        # ─── 2. Sparse Lexical Search Processing (Elasticsearch BM25) ────────
        bm25_shot_scores = {}
        bm25_data = {}

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

        # Apply Text filter
        if text_filter_ranges is not None:
            filtered = {}
            for pid, score in final_scores.items():
                payload = all_payloads.get(pid, {})
                vid = payload.get("video_id")
                rep_frame = payload.get("representative_frame", payload.get("keyframe_idx", 0))
                if self._frame_in_text_filter_ranges(vid, rep_frame, text_filter_ranges) > 0.0:
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

            item = {
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
            }
            if translated_query_str:
                item["translated_query"] = translated_query_str

            results.append(item)

            if len(results) >= limit:
                break

        return results


    def fused_search(self, query: str, model_names: list = None, objects: list = None,
                     top_k: int = 1000, score_threshold: float = 0.0,
                     limit: int = 100, ocr_query=None, audio_query=None,
                     keyframe_weight: float = None, caption_weight: float = None,
                     text_filters = None, group_by_shot: bool = False):
        """
        Fused search: combine keyframe embedding search and caption embedding search.
        For each keyframe in the union of both result sets:
          - If matched by both modalities: combine with multimodal synergy boost (>= max(kf, cap))
          - If matched by only one modality: preserve original normalized score (not halved)
        Caption search expands shot results to ALL keyframes in that shot's range.
        """
        from config import KEYFRAME_SEARCH_WEIGHT, CAPTION_SEARCH_WEIGHT

        w_kf = keyframe_weight if keyframe_weight is not None else KEYFRAME_SEARCH_WEIGHT
        w_cap = caption_weight if caption_weight is not None else CAPTION_SEARCH_WEIGHT

        logger.info(f"Performing fused search for: '{query}' (kf_weight={w_kf}, cap_weight={w_cap}, text_filters={text_filters})")

        # Pre-compute text_filter_ranges once for both modalities
        text_filter_ranges = None
        if text_filters:
            text_filter_ranges = self._get_text_filter_ranges(text_filters, model_names=model_names)
            if not text_filter_ranges:
                logger.info("No matches for text_filters in fused_search. Returning empty results.")
                return []

        # 1 & 2. Run keyframe search and caption search concurrently in parallel
        def _run_kf():
            return self.semantic_search(
                query, model_names=model_names, objects=objects,
                top_k=top_k, score_threshold=0.0,
                limit=top_k, group_by_shot=False,
                ocr_query=ocr_query, audio_query=audio_query,
                text_filters=text_filter_ranges
            )

        def _run_cap():
            return self.caption_search(
                query, top_k=top_k, score_threshold=0.0,
                limit=top_k, ocr_query=ocr_query, audio_query=audio_query,
                objects=objects,
                text_filters=text_filter_ranges
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            kf_future = executor.submit(_run_kf)
            cap_future = executor.submit(_run_cap)
            kf_results = kf_future.result()
            cap_results = cap_future.result()

        cap_translated_query = next((r["translated_query"] for r in cap_results if "translated_query" in r), None)

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
            in_kf = key in kf_score_map
            in_cap = key in cap_score_map
            kf_s = kf_norm.get(key, 0.0)
            cap_s = cap_norm.get(key, 0.0)

            if in_kf and in_cap:
                # Keyframe xuất hiện ở cả 2 phương thức: cộng hưởng đa phương thức (multimodal synergy)
                # Đảm bảo điểm kết hợp không bị kéo tụt xuống thấp hơn điểm của từng nguồn đơn lẻ
                total_w = w_kf + w_cap
                weighted_avg = (w_kf * kf_s + w_cap * cap_s) / total_w if total_w > 0 else (kf_s + cap_s) / 2.0
                base_score = max(weighted_avg, max(kf_s, cap_s))
                # Điểm thưởng cộng hưởng khi cả 2 nguồn đều xác nhận
                synergy_bonus = 0.15 * min(kf_s, cap_s)
                fused = min(1.0, base_score + synergy_bonus)
            elif in_kf:
                # Chỉ xuất hiện ở keyframe embedding search -> giữ nguyên điểm, không bị chia đôi
                fused = kf_s
            else:
                # Chỉ xuất hiện ở caption search -> giữ nguyên điểm, không bị chia đôi
                fused = cap_s

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

            fused_item = {
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
            }
            if cap_translated_query:
                fused_item["translated_query"] = cap_translated_query

            fused_results.append(fused_item)

        fused_results.sort(key=lambda x: x["score"], reverse=True)

        if group_by_shot:
            shot_dict = {}
            for item in fused_results:
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
                    "display_frames": items,
                    "keyframe_index": anchor["keyframe_index"],
                    "shot_start_frame": start,
                    "shot_end_frame": end,
                    "result_type": "fused",
                    "ocr_text": anchor.get("ocr_text", ""),
                    "audio_text": anchor.get("audio_text", ""),
                    "caption_text": anchor.get("caption_text", ""),
                    "best_aspect": anchor.get("best_aspect", ""),
                    "best_aspect_caption": anchor.get("best_aspect_caption", ""),
                })

            grouped_results.sort(key=lambda x: x["score"], reverse=True)
            return grouped_results[:limit]

        return fused_results[:limit]

if __name__ == "__main__":
    system = RetrievalSystem(init_text_encoders=False)
    # results = system.semantic_search("A person riding a horse on a beach.")
    # system.ingest_shot_caption_embedding()
    system.ingest_all_vision_embedding(max_workers=12)
