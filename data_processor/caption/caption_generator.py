import os
import sys
import json
import glob
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from .prompt import build_shot_prompt, parse_aspects

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("caption_generator")


class ShotCaptionGenerator:
    def __init__(
        self,
        data_dir: str = "data",
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_id: Optional[str] = None,
        concurrency: int = 5,
        ocr_sources: Optional[List[str]] = None,
    ):
        load_dotenv()
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "caption"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.project_id = project_id or os.environ.get("PROJECT_ID", "")
        self.location = location or os.environ.get("VERTEX_LOCATION", "global")
        self.model_id = model_id or os.environ.get("MODEL_ID", "gemini-2.5-flash-lite")
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)

        self.ocr_sources = ocr_sources or ["V6", "VL1.6"]

        logger.info(
            f"Initializing Gemini Client (Project: '{self.project_id}', "
            f"Location: '{self.location}', Model: '{self.model_id}', Concurrency: {self.concurrency})"
        )
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

        # Load video metadata (FPS)
        self.metadata = self._load_video_metadata()
        # Load all shot boundaries
        self.shots_data = self._load_all_shots()

    def _load_video_metadata(self) -> Dict[str, dict]:
        meta_file = self.data_dir / "video_metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning(f"Video metadata file '{meta_file}' not found. Defaulting FPS to 25.0.")
        return {}

    def _load_all_shots(self) -> Dict[str, List[List[int]]]:
        shot_dir = self.data_dir / "shot"
        shots_map = {}
        if not shot_dir.exists():
            logger.warning(f"Shot directory '{shot_dir}' does not exist.")
            return shots_map

        for fpath in glob.glob(str(shot_dir / "*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for video_key, boundaries in data.items():
                        v_id = video_key.split(".")[0]
                        shots_map[v_id] = sorted(boundaries, key=lambda x: x[0])
            except Exception as e:
                logger.error(f"Error loading shot file '{fpath}': {e}")
        logger.info(f"Loaded shot boundaries for {len(shots_map)} videos.")
        return shots_map

    def _get_video_fps(self, video_id: str) -> float:
        for key in (f"{video_id}.mp4", video_id):
            if key in self.metadata and "fps" in self.metadata[key]:
                return float(self.metadata[key]["fps"])
        return 25.0

    def _get_available_keyframes(self, video_id: str) -> List[int]:
        kf_dir = self.data_dir / "keyframe" / video_id
        if not kf_dir.exists():
            return []
        kfs = []
        for fname in os.listdir(kf_dir):
            if fname.startswith("keyframe_") and fname.endswith(".webp"):
                try:
                    idx = int(fname.split("_")[1].split(".")[0])
                    kfs.append(idx)
                except ValueError:
                    continue
        return sorted(kfs)

    def _sample_keyframes_for_shot(
        self, available_kfs: List[int], start_frame: int, end_frame: int
    ) -> List[int]:
        in_shot = [kf for kf in available_kfs if start_frame <= kf <= end_frame]
        if not in_shot:
            if not available_kfs:
                return []
            # Find closest keyframe to middle
            mid = (start_frame + end_frame) // 2
            closest = min(available_kfs, key=lambda x: abs(x - mid))
            return [closest]
        if len(in_shot) == 1:
            return in_shot
        if len(in_shot) == 2:
            return in_shot
        # Sample 3 keyframes: start, middle, end of shot
        return [in_shot[0], in_shot[len(in_shot) // 2], in_shot[-1]]

    def _load_transcript_for_range(
        self, video_id: str, t_start: float, t_end: float
    ) -> str:
        trans_file = self.data_dir / "transcript" / f"{video_id}.json"
        if not trans_file.exists():
            return ""
        try:
            with open(trans_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                segments = data.get("segments", [])
                matching_texts = [
                    seg.get("text", "").strip()
                    for seg in segments
                    if seg.get("end", 0) >= t_start and seg.get("start", 0) <= t_end
                ]
                return " ".join([t for t in matching_texts if t])
        except Exception as e:
            logger.debug(f"Could not load transcript for {video_id}: {e}")
            return ""

    def _load_ocr_for_keyframes(
        self, video_id: str, keyframes: List[int]
    ) -> List[str]:
        texts = set()
        ocr_root = self.data_dir / "ocr"
        for source in self.ocr_sources:
            for kf in keyframes:
                ocr_file = ocr_root / source / video_id / f"keyframe_{kf}.json"
                if not ocr_file.exists():
                    continue
                try:
                    with open(ocr_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # V6 uses 'rec_texts'
                        rec = data.get("rec_texts", [])
                        for t in rec:
                            t_clean = t.strip()
                            if t_clean and len(t_clean) > 1:
                                texts.add(t_clean)
                        # VL1.6 uses 'parsing_res_list'
                        parsing = data.get("parsing_res_list", [])
                        for p in parsing:
                            t = p.get("text", "").strip()
                            if t and len(t) > 1:
                                texts.add(t)
                except Exception:
                    continue
        return list(texts)

    async def _caption_single_shot(
        self,
        video_id: str,
        shot_idx: int,
        start_frame: int,
        end_frame: int,
        fps: float,
        available_kfs: List[int],
    ) -> Optional[dict]:
        t_start = start_frame / fps
        t_end = end_frame / fps

        sampled_kfs = self._sample_keyframes_for_shot(available_kfs, start_frame, end_frame)
        if not sampled_kfs:
            logger.warning(f"[{video_id}] Shot {shot_idx} has no available keyframes. Skipping.")
            return None

        # Read keyframe image bytes
        image_parts = []
        for kf in sampled_kfs:
            kf_path = self.data_dir / "keyframe" / video_id / f"keyframe_{kf}.webp"
            if not kf_path.exists():
                continue
            with open(kf_path, "rb") as f:
                img_bytes = f.read()
            image_parts.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/webp")
            )

        if not image_parts:
            logger.warning(f"[{video_id}] Shot {shot_idx}: Failed to load image bytes for {sampled_kfs}.")
            return None

        # Gather OCR and audio
        shot_ocr = self._load_ocr_for_keyframes(video_id, sampled_kfs)
        shot_audio = self._load_transcript_for_range(video_id, t_start, t_end)

        # Build prompt
        prompt_text = build_shot_prompt(
            t_start=t_start,
            t_end=t_end,
            ocr_texts=shot_ocr,
            audio_transcript=shot_audio
        )

        contents = image_parts + [prompt_text]
        config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            temperature=0.2,
        )

        # Call API with semaphore and retry logic
        max_retries = 4
        base_delay = 2.0
        caption_text = ""

        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    response = await self.client.aio.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config=config,
                    )
                    caption_text = response.text or ""
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"[{video_id} - Shot {shot_idx}] Rate limit hit. Retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"[{video_id} - Shot {shot_idx}] Error during generation: {e}")
                        if attempt == max_retries - 1:
                            return None
                        await asyncio.sleep(base_delay)

        if not caption_text.strip():
            logger.warning(f"[{video_id} - Shot {shot_idx}] Empty caption generated.")
            return None

        parsed = parse_aspects(caption_text)

        return {
            "shot_idx": shot_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_sec": round(t_start, 2),
            "end_sec": round(t_end, 2),
            "sampled_keyframes": sampled_kfs,
            "ocr_detected": shot_ocr,
            "audio_transcript": shot_audio,
            "caption_text": caption_text.strip(),
            "aspects": parsed,
        }

    async def process_video(
        self,
        video_id: str,
        overwrite: bool = False,
        max_shots: Optional[int] = None
    ) -> bool:
        out_file = self.output_dir / f"{video_id}.json"
        existing_shots_map: Dict[int, dict] = {}

        if out_file.exists() and not overwrite:
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    for s in cached_data.get("shots", []):
                        existing_shots_map[s["shot_idx"]] = s
            except Exception as e:
                logger.warning(f"Could not read existing caption file '{out_file}': {e}")

        shots = self.shots_data.get(video_id, [])
        if not shots:
            logger.warning(f"No shot boundaries found for video '{video_id}'. Skipping.")
            return False

        if max_shots is not None and max_shots > 0:
            shots = shots[:max_shots]

        # Determine which shots need to be processed
        if overwrite:
            shots_to_run = list(enumerate(shots))
        else:
            shots_to_run = [
                (idx, s) for idx, s in enumerate(shots)
                if idx not in existing_shots_map
            ]

        if not shots_to_run:
            logger.info(f"All {len(shots)} shots for '{video_id}' are already captioned in '{out_file}'. Skipping (use --overwrite to re-run).")
            return True

        fps = self._get_video_fps(video_id)
        available_kfs = self._get_available_keyframes(video_id)
        if not available_kfs:
            logger.warning(f"No keyframes found for video '{video_id}'. Skipping.")
            return False

        if existing_shots_map and not overwrite:
            logger.info(
                f"Resuming '{video_id}': processing {len(shots_to_run)} remaining shots "
                f"({len(existing_shots_map)} already cached, total {len(shots)})."
            )
        else:
            logger.info(f"Processing '{video_id}': {len(shots_to_run)} shots (FPS: {fps:.2f}, Keyframes: {len(available_kfs)})")

        tasks = [
            self._caption_single_shot(
                video_id=video_id,
                shot_idx=idx,
                start_frame=s[0],
                end_frame=s[1],
                fps=fps,
                available_kfs=available_kfs,
            )
            for idx, s in shots_to_run
        ]

        shot_results = await async_tqdm.gather(*tasks, desc=f"🎬 [{video_id}]", unit="shot")
        new_valid = [r for r in shot_results if r is not None]

        # Merge with existing shots
        merged_shots_dict = dict(existing_shots_map) if not overwrite else {}
        for r in new_valid:
            merged_shots_dict[r["shot_idx"]] = r

        final_shots = sorted(merged_shots_dict.values(), key=lambda x: x["shot_idx"])

        output_payload = {
            "video_id": video_id,
            "total_shots": len(self.shots_data.get(video_id, shots)),
            "captioned_shots": len(final_shots),
            "fps": fps,
            "shots": final_shots
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(final_shots)} shot captions for '{video_id}' to '{out_file}'.")
        return True


    async def process_all_videos(
        self,
        video_ids: Optional[List[str]] = None,
        overwrite: bool = False,
        max_shots_per_video: Optional[int] = None,
        limit_videos: Optional[int] = None,
    ):
        if not video_ids:
            # Find all videos with shot boundaries and keyframes
            all_kf_videos = {
                d.name for d in (self.data_dir / "keyframe").iterdir() if d.is_dir()
            }
            video_ids = sorted(list(set(self.shots_data.keys()).intersection(all_kf_videos)))

        if limit_videos and limit_videos > 0:
            video_ids = video_ids[:limit_videos]

        logger.info(f"Starting batch caption generation for {len(video_ids)} videos...")

        for vid in tqdm(video_ids, desc="🎥 Videos", unit="video"):
            try:
                await self.process_video(
                    video_id=vid,
                    overwrite=overwrite,
                    max_shots=max_shots_per_video
                )
            except Exception as e:
                logger.error(f"Failed to process video '{vid}': {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Aspect Video Shot Caption Generator")
    parser.add_argument("--video_id", type=str, default=None, help="Process a single video ID (e.g. L21_V001)")
    parser.add_argument("--all", action="store_true", help="Process all videos in data/")
    parser.add_argument("--max_videos", type=int, default=None, help="Limit number of videos to process")
    parser.add_argument("--max_shots", type=int, default=None, help="Limit number of shots per video (for testing)")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent API calls (default: 5)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caption files")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory (default: data)")

    args = parser.parse_args()

    generator = ShotCaptionGenerator(
        data_dir=args.data_dir,
        concurrency=args.concurrency
    )

    if args.video_id:
        asyncio.run(
            generator.process_video(
                video_id=args.video_id,
                overwrite=args.overwrite,
                max_shots=args.max_shots
            )
        )
    elif args.all:
        asyncio.run(
            generator.process_all_videos(
                overwrite=args.overwrite,
                max_shots_per_video=args.max_shots,
                limit_videos=args.max_videos
            )
        )
    else:
        # Default test: run 1 shot of L21_V001 to verify
        test_video = "L21_V001"
        logger.info(f"No specific flag provided. Running test on '{test_video}' (first 2 shots)...")
        asyncio.run(
            generator.process_video(
                video_id=test_video,
                overwrite=True,
                max_shots=2
            )
        )


if __name__ == "__main__":
    main()
