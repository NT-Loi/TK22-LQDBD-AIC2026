#!/usr/bin/env python3
"""
AIC Frame Retrieval Agent — Special Detail + Temporal Expansion
================================================================
Coding Agent là LLM lõi duy nhất.
Pipeline: Parse → Anchor Search → Deduplicate → Contact Sheet → Temporal Expansion → Suspects List

Usage:
    python retrieval_agent.py audit
    python retrieval_agent.py search --text "..." [--ocr "..."] [--audio "..."] [--limit 100] [--output out.json]
    python retrieval_agent.py sheet  --input results.json [--limit 20] [--columns 5] [--output sheet.jpg]
    python retrieval_agent.py context --video-id VID --keyframe-index IDX [--neighbors 12] [--columns 4]
    python retrieval_agent.py keyframes --video-id VID
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import requests

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[4]   # project root
API_BASE   = "http://127.0.0.1:8000"
QDRANT_URL = "http://127.0.0.1:6333"
ES_URL     = "http://127.0.0.1:9200"
DATA_DIR   = BASE_DIR / "data"
ARTIFACTS  = BASE_DIR / "artifacts" / "aic_agent"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

THUMB_W, THUMB_H = 320, 180   # contact sheet thumbnail size
LABEL_H          = 40          # pixels for label bar under each thumb

# ── Helpers ────────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    """Format seconds to MM:SS.d"""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m:02d}:{s:04.1f}"


def _get_fps(video_id: str) -> float:
    try:
        r = requests.get(f"{API_BASE}/api/video_info/{video_id}", timeout=5)
        if r.ok:
            return float(r.json().get("fps", 25.0))
    except Exception:
        pass
    return 25.0


def _keyframe_path(video_id: str, frame_idx: int) -> Path | None:
    """Return path to keyframe image, extracting from video if .webp missing."""
    webp = DATA_DIR / "keyframe" / video_id / f"keyframe_{frame_idx}.webp"
    if webp.exists():
        return webp

    # Fallback: extract from .mp4 with OpenCV
    mp4 = DATA_DIR / "video" / f"{video_id}.mp4"
    if not mp4.exists():
        return None

    try:
        import cv2
        cap = cv2.VideoCapture(str(mp4))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        webp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(webp), frame, [cv2.IMWRITE_WEBP_QUALITY, 85])
        return webp
    except Exception:
        return None


def _render_sheet(entries: list, output_path: Path, columns: int = 5) -> Path:
    """Render a contact sheet from a list of {video_id, keyframe_index, score?, ...}."""
    from PIL import Image, ImageDraw, ImageFont

    rows = math.ceil(len(entries) / columns)
    cell_w = THUMB_W
    cell_h = THUMB_H + LABEL_H
    img = Image.new("RGB", (cell_w * columns, cell_h * rows), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    for i, entry in enumerate(entries):
        col = i % columns
        row = i // columns
        x0, y0 = col * cell_w, row * cell_h

        vid   = entry.get("video_id", "?")
        fidx  = entry.get("keyframe_index", 0)
        score = entry.get("score", 0.0)
        fps   = entry.get("fps", _get_fps(vid))
        ts    = fidx / fps

        kp = _keyframe_path(vid, fidx)
        if kp:
            try:
                thumb = Image.open(kp).resize((THUMB_W, THUMB_H), Image.LANCZOS)
                img.paste(thumb, (x0, y0))
            except Exception:
                pass
        else:
            draw.rectangle([x0, y0, x0 + THUMB_W, y0 + THUMB_H], fill=(60, 20, 20))
            draw.text((x0 + 8, y0 + THUMB_H // 2 - 8), "IMAGE MISSING", fill=(200, 80, 80), font=font)

        # Label bar
        lbl_y = y0 + THUMB_H
        draw.rectangle([x0, lbl_y, x0 + cell_w, lbl_y + LABEL_H], fill=(20, 20, 20))
        line1 = f"#{i+1} {vid}  f={fidx}"
        line2 = f"t={ts:.2f}s ({_fmt(ts)})  sc={score:.4f}"
        draw.text((x0 + 4, lbl_y + 2),  line1, fill=(255, 220, 80),  font=font)
        draw.text((x0 + 4, lbl_y + 20), line2, fill=(180, 220, 255), font=font)

    img.save(str(output_path), quality=90)
    return output_path


def _deduplicate(results: list, min_gap_seconds: float = 180.0) -> list:
    """Keep at most one result per video unless frames are far apart (>= min_gap_seconds)."""
    seen: dict[str, list[float]] = {}   # video_id → [timestamp_seconds]
    out = []
    for r in results:
        vid   = r.get("video_id", "")
        fidx  = r.get("keyframe_index", 0)
        fps   = r.get("fps", 25.0)
        ts    = fidx / fps

        if vid not in seen:
            seen[vid] = [ts]
            out.append(r)
        else:
            if all(abs(ts - prev) >= min_gap_seconds for prev in seen[vid]):
                seen[vid].append(ts)
                out.append(r)
    return out


# ── Commands ───────────────────────────────────────────────────────────────

def command_audit() -> dict:
    counts = {
        "videos":       len(list((DATA_DIR / "video").glob("*.mp4")))      if (DATA_DIR / "video").exists()      else 0,
        "keyframes":    sum(1 for _ in (DATA_DIR / "keyframe").rglob("*.webp")) if (DATA_DIR / "keyframe").exists() else 0,
        "ocr_files":    sum(1 for _ in (DATA_DIR / "ocr").rglob("*.txt"))  if (DATA_DIR / "ocr").exists()        else 0,
        "shot_files":   len(list((DATA_DIR / "shot").glob("*.json")))       if (DATA_DIR / "shot").exists()       else 0,
        "transcript":   len(list((DATA_DIR / "transcript").glob("*.json"))) if (DATA_DIR / "transcript").exists() else 0,
    }
    services = {}
    for name, url in [("api", f"{API_BASE}/api/models"),
                      ("qdrant", f"{QDRANT_URL}/collections"),
                      ("elasticsearch", f"{ES_URL}/_cluster/health")]:
        try:
            r = requests.get(url, timeout=5)
            services[name] = {"ok": r.ok, "status": r.status_code}
        except Exception as e:
            services[name] = {"ok": False, "error": str(e)}

    result = {"data_dir": str(DATA_DIR), "counts": counts, "services": services}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_search(text: list[str], ocr: str | None, audio: str | None,
                   limit: int, output: str | None,
                   min_gap_sec: float = 180.0, top_n: int = 20) -> list:
    """Search and return deduplicated top-N distinct candidates."""
    payload: dict = {"limit": max(limit, 100)}

    if text:
        payload["text_queries"] = [{"text": t} for t in text]
    if ocr:
        payload["ocr_query"] = ocr
    if audio:
        payload["audio"] = [{"text": audio, "level": "video"}]

    try:
        r = requests.post(f"{API_BASE}/search", json=payload, timeout=60)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        print(f"[ERROR] Search failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Flatten grouped results
    flat = []
    for item in raw:
        if "frames" in item:
            flat.extend(item["frames"])
        else:
            flat.append(item)

    # Sort by score desc, deduplicate, take top-N
    flat.sort(key=lambda x: x.get("score", 0), reverse=True)
    deduped = _deduplicate(flat, min_gap_seconds=min_gap_sec)
    top = deduped[:top_n]

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(top, f, indent=2, ensure_ascii=False)
        print(out_path)
    else:
        print(json.dumps(top, indent=2, ensure_ascii=False))

    return top


def command_sheet(input_file: str, limit: int, columns: int, output: str | None) -> Path:
    """Render a contact sheet from a JSON results file."""
    with open(input_file, encoding="utf-8") as f:
        entries = json.load(f)

    entries = entries[:limit]
    stem = Path(input_file).stem
    out_path = Path(output) if output else ARTIFACTS / f"sheet_{stem}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _render_sheet(entries, out_path, columns=columns)
    print(out_path)
    return out_path


def command_context(video_id: str, keyframe_index: int, neighbors: int, columns: int, output: str | None) -> Path:
    """Show neighboring keyframes around a given frame (temporal context)."""
    # Get all keyframes for this video from the API
    try:
        r = requests.get(f"{API_BASE}/api/video_keyframes/{video_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        all_kf = [kf["keyframe_index"] for kf in data.get("keyframes", [])]
        fps = data.get("fps", 25.0)
    except Exception as e:
        print(f"[ERROR] Cannot fetch keyframes for {video_id}: {e}", file=sys.stderr)
        sys.exit(1)

    all_kf.sort()
    if keyframe_index not in all_kf:
        # find closest
        closest = min(all_kf, key=lambda x: abs(x - keyframe_index))
        keyframe_index = closest

    idx = all_kf.index(keyframe_index)
    lo = max(0, idx - neighbors)
    hi = min(len(all_kf) - 1, idx + neighbors)
    window = all_kf[lo:hi + 1]

    entries = [{"video_id": video_id, "keyframe_index": kf, "fps": fps, "score": 0.0} for kf in window]

    stem = f"ctx_{video_id}_{keyframe_index}"
    out_path = Path(output) if output else ARTIFACTS / f"{stem}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _render_sheet(entries, out_path, columns=columns)
    print(out_path)
    return out_path


def command_keyframes(video_id: str) -> list:
    """List all keyframe indices for a video."""
    try:
        r = requests.get(f"{API_BASE}/api/video_keyframes/{video_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        kfs = sorted([kf["keyframe_index"] for kf in data.get("keyframes", [])])
        fps = data.get("fps", 25.0)
        print(f"Video: {video_id}  FPS: {fps}  Total keyframes: {len(kfs)}")
        for kf in kfs:
            ts = kf / fps
            print(f"  frame {kf:6d}  t={ts:8.2f}s  ({_fmt(ts)})")
        return kfs
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


# ── Suspects List Printer ──────────────────────────────────────────────────

def print_suspects(results: list, query_used: str, special_detail_position: str = "?",
                   direction: str = "?") -> None:
    """Print the formatted Suspects List as described in the implementation plan."""
    print()
    print("═" * 70)
    print("ANCHOR SEARCH KẾT THÚC")
    print(f"Query đã dùng: \"{query_used}\"")
    print(f"Special detail vị trí: {special_detail_position}")
    print(f"Hướng mở rộng: {direction}")
    print("═" * 70)
    print()
    print("📋 TOP ANCHOR FRAMES (khớp với special detail):")
    print(f"{'#':>4}  {'video_id':<12}  {'frame_idx':>10}  {'time':>9}  {'score':>7}")
    print("-" * 55)
    for i, r in enumerate(results):
        vid   = r.get("video_id", "?")
        fidx  = r.get("keyframe_index", 0)
        fps   = r.get("fps", 25.0)
        score = r.get("score", 0.0)
        ts    = fidx / fps
        print(f"{i+1:>4}  {vid:<12}  {fidx:>10}  {_fmt(ts):>9}  {score:>7.4f}")
    print()
    print("ℹ️  Dùng 'context' để mở rộng temporal từ bất kỳ frame nào ở trên.")
    print("ℹ️  Để tìm lại: chọn chi tiết khác, sửa query, hoặc thêm filter.")
    print("═" * 70)


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AIC Frame Retrieval Agent")
    sub = p.add_subparsers(dest="command", required=True)

    # audit
    sub.add_parser("audit", help="Check data coverage and service health")

    # search
    s = sub.add_parser("search", help="Anchor search with deduplication")
    s.add_argument("--text",   action="append", default=[], dest="text",
                   help="Text query (can repeat for temporal search)")
    s.add_argument("--ocr",    default=None,    help="OCR text filter")
    s.add_argument("--audio",  default=None,    help="Audio/transcript filter")
    s.add_argument("--limit",  type=int, default=100, help="Raw result limit before dedup")
    s.add_argument("--top",    type=int, default=20,  help="Top-N distinct after dedup")
    s.add_argument("--min-gap", type=float, default=180.0,
                   help="Min gap in seconds between same-video results (default: 180)")
    s.add_argument("--output", default=None,    help="Save results to JSON file")
    # For suspects list printing
    s.add_argument("--position", default="?",  help="Special detail position, e.g. '3/5'")
    s.add_argument("--direction", default="?", help="BACKWARD | BIDIRECTIONAL | FORWARD")

    # sheet
    sh = sub.add_parser("sheet", help="Render contact sheet from results JSON")
    sh.add_argument("--input",   required=True, help="Input JSON file")
    sh.add_argument("--limit",   type=int, default=20,  help="Max frames to show")
    sh.add_argument("--columns", type=int, default=5,   help="Columns in sheet")
    sh.add_argument("--output",  default=None, help="Output image path")

    # context
    ctx = sub.add_parser("context", help="Show temporal context around a frame")
    ctx.add_argument("--video-id",       required=True)
    ctx.add_argument("--keyframe-index", required=True, type=int)
    ctx.add_argument("--neighbors",      type=int, default=12,
                     help="Number of keyframes either side (default: 12)")
    ctx.add_argument("--columns",        type=int, default=4)
    ctx.add_argument("--output",         default=None)

    # keyframes
    kf = sub.add_parser("keyframes", help="List all keyframe indices for a video")
    kf.add_argument("--video-id", required=True)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "audit":
        command_audit()

    elif args.command == "search":
        results = command_search(
            text=args.text,
            ocr=args.ocr,
            audio=args.audio,
            limit=args.limit,
            output=args.output,
            min_gap_sec=args.min_gap,
            top_n=args.top,
        )
        if not args.output:
            # Print suspects list format if not saving to file
            query_used = " | ".join(args.text) if args.text else (args.ocr or args.audio or "")
            print_suspects(results, query_used=query_used,
                           special_detail_position=args.position,
                           direction=args.direction)

    elif args.command == "sheet":
        command_sheet(
            input_file=args.input,
            limit=args.limit,
            columns=args.columns,
            output=args.output,
        )

    elif args.command == "context":
        command_context(
            video_id=args.video_id,
            keyframe_index=args.keyframe_index,
            neighbors=args.neighbors,
            columns=args.columns,
            output=args.output,
        )

    elif args.command == "keyframes":
        command_keyframes(video_id=args.video_id)


if __name__ == "__main__":
    main()

