import os
import csv
import bisect
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

try:
    from config import MAP_DIRS
except ImportError:
    MAP_DIRS = [Path("data/maps")]


class PTSMapper:
    """
    Manages PTS (Presentation Time Stamp) mappings for videos.
    Uses piecewise linear interpolation between map keyframe anchors (spaced ~2s apart)
    to compute exact milliseconds (pts_time * 1000) for any frame_idx across both CFR and VFR videos.
    """

    def __init__(self, map_dirs: Optional[List[Path]] = None):
        self.map_dirs = map_dirs or MAP_DIRS
        # maps[video_stem] = (frames_list, pts_list, nominal_fps)
        self.maps: Dict[str, Tuple[List[int], List[float], float]] = {}
        self._loaded = False

    def load_maps(self, force_reload: bool = False):
        if self._loaded and not force_reload:
            return

        loaded_count = 0
        for mdir in self.map_dirs:
            pdir = Path(mdir)
            if not pdir.is_dir():
                continue
            for csv_file in pdir.glob("*.csv"):
                stem = csv_file.stem
                try:
                    frames = []
                    pts_list = []
                    nominal_fps = 25.0
                    with open(csv_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            f_idx = int(row["frame_idx"])
                            p_time = float(row["pts_time"])
                            frames.append(f_idx)
                            pts_list.append(p_time)
                            if "fps" in row and row["fps"]:
                                try:
                                    nominal_fps = float(row["fps"])
                                except (ValueError, TypeError):
                                    pass

                    if frames:
                        val = (frames, pts_list, nominal_fps)
                        self.maps[stem] = val
                        # Also register variants (e.g. N001-V001 and N001_V001)
                        if "-" in stem:
                            self.maps[stem.replace("-", "_")] = val
                        elif "_" in stem:
                            self.maps[stem.replace("_", "-")] = val
                        loaded_count += 1
                except Exception as e:
                    logger.warning(f"Error reading map CSV {csv_file}: {e}")

        self._loaded = True
        logger.info(f"PTSMapper loaded {loaded_count} video maps ({len(self.maps)} lookup keys).")

    def _get_map_entry(self, video_id: str) -> Optional[Tuple[List[int], List[float], float]]:
        if not self._loaded:
            self.load_maps()

        raw = os.path.splitext(os.path.basename(video_id))[0].strip()
        if raw in self.maps:
            return self.maps[raw]

        variants = [
            raw.replace("-", "_"),
            raw.replace("_", "-"),
            raw.upper(),
            raw.lower(),
        ]
        for v in variants:
            if v in self.maps:
                return self.maps[v]
        return None

    def get_pts_time(self, video_id: str, frame_idx: int, default_fps: float = 25.0) -> float:
        """
        Returns the presentation time in seconds for a given frame_idx.
        If the video has a map, interpolates between adjacent anchors.
        Otherwise falls back to frame_idx / default_fps.
        """
        entry = self._get_map_entry(video_id)
        if not entry:
            return frame_idx / (default_fps if default_fps > 0 else 25.0)

        frames, pts_list, nominal_fps = entry
        if not frames:
            return frame_idx / (default_fps if default_fps > 0 else 25.0)

        idx = bisect.bisect_right(frames, frame_idx) - 1
        if idx < 0:
            return pts_list[0]
        if idx >= len(frames) - 1:
            eff_fps = nominal_fps if nominal_fps > 0 else default_fps
            return pts_list[-1] + (frame_idx - frames[-1]) / eff_fps

        f1, f2 = frames[idx], frames[idx + 1]
        p1, p2 = pts_list[idx], pts_list[idx + 1]

        if f2 == f1:
            return p1

        return p1 + (frame_idx - f1) / (f2 - f1) * (p2 - p1)

    def get_time_ms(self, video_id: str, frame_idx: int, default_fps: float = 25.0) -> int:
        """Returns the PTS in milliseconds rounded to the nearest integer."""
        pts_sec = self.get_pts_time(video_id, frame_idx, default_fps)
        return max(0, round(pts_sec * 1000))

    def annotate_result_item(self, item: Dict[str, Any], default_fps: float = 25.0) -> Dict[str, Any]:
        """
        Enriches a search result item (and its nested frames/sequences) with accurate fps and timeMs.
        """
        vid = str(item.get("video_id") or item.get("videoId") or "")
        fps = float(item.get("fps") or default_fps)
        if fps <= 0:
            fps = 25.0

        entry = self._get_map_entry(vid)
        if entry:
            _, _, map_fps = entry
            if map_fps > 0:
                fps = map_fps

        item["fps"] = fps

        kf_idx = item.get("keyframe_index")
        if kf_idx is None:
            kf_idx = item.get("frameId")
        if kf_idx is not None:
            try:
                kf_idx_int = int(kf_idx)
                item["timeMs"] = self.get_time_ms(vid, kf_idx_int, fps)
            except (ValueError, TypeError):
                pass

        # Nested collections
        for sub_key in ("frames", "display_frames", "temporal_sequence"):
            if sub_key in item and isinstance(item[sub_key], list):
                for sub_item in item[sub_key]:
                    if isinstance(sub_item, dict):
                        sub_vid = str(sub_item.get("video_id") or sub_item.get("videoId") or vid)
                        sub_item["fps"] = fps
                        sub_idx = sub_item.get("keyframe_index")
                        if sub_idx is None:
                            sub_idx = sub_item.get("frameId")
                        if sub_idx is not None:
                            try:
                                sub_item["timeMs"] = self.get_time_ms(sub_vid, int(sub_idx), fps)
                            except (ValueError, TypeError):
                                pass

        return item


# Global default instance
_default_mapper = PTSMapper()


def get_video_pts_time(video_id: str, frame_idx: int, default_fps: float = 25.0) -> float:
    return _default_mapper.get_pts_time(video_id, frame_idx, default_fps)


def get_video_time_ms(video_id: str, frame_idx: int, default_fps: float = 25.0) -> int:
    return _default_mapper.get_time_ms(video_id, frame_idx, default_fps)


def annotate_result_item(item: Dict[str, Any], default_fps: float = 25.0) -> Dict[str, Any]:
    return _default_mapper.annotate_result_item(item, default_fps)
