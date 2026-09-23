#!/usr/bin/env python3
"""
query_runner.py - Fast CLI query execution runner for AIC 2026 video retrieval.
Supports Textual KIS, Visual Q&A candidate search, and TRAKE temporal alignment.
"""

import argparse
import sys
import os
import csv
import json
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path so retrieval_system can be imported cleanly
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(description="AIC 2026 Video Retrieval Query Runner")
    parser.add_argument("--type", choices=["kis", "qa", "trake"], required=True,
                        help="Query type: 'kis' (Textual KIS), 'qa' (Visual Q&A), or 'trake' (Temporal Alignment)")
    parser.add_argument("--query", type=str, default="",
                        help="Visual scene or event description query in Vietnamese/English")
    parser.add_argument("--events", nargs="+", default=[],
                        help="Ordered sub-event descriptions for TRAKE (e.g. --events 'chạy đà' 'giậm nhảy' 'tiếp đất')")
    parser.add_argument("--ocr", type=str, default=None,
                        help="Optional on-screen text query for OCR matching in Vietnamese")
    parser.add_argument("--audio", type=str, default=None,
                        help="Optional spoken words for audio transcript matching in Vietnamese")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Embedding models to query (default: all initialized, e.g. SigLIP SigLIP2 CLIP_H14)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum number of candidates to return (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (CSV or JSON) to save results")
    parser.add_argument("--subqueries", nargs="+", default=[],
                        help="Explicit list of atomic visual sub-clauses for multi-query neighborhood fusion")
    parser.add_argument("--decompose", action="store_true", default=True,
                        help="Automatically decompose compound queries into atomic visual sub-clauses (default: True)")
    parser.add_argument("--neighborhood-window", type=int, default=250,
                        help="Temporal frame distance to aggregate complementary sub-clauses in the same scene (default: 250)")
    parser.add_argument("--dedup-window", type=int, default=15,
                        help="Temporal frame window within same video to cluster near-duplicate frames into one candidate (default: 15, 0 to disable)")
    parser.add_argument("--mode", choices=["keyframe", "caption", "both"], default="keyframe",
                        help="Search target mode: 'keyframe' (visual embeddings), 'caption' (hybrid caption), or 'both' (fused)")
    parser.add_argument("--log-trace", action="store_true", default=True,
                        help="Print structured execution and reasoning trace")
    return parser.parse_args()


def print_search_trace(args, total_results):
    print("\n" + "=" * 60)
    print("🔍 SEARCH STRATEGY & EXECUTION TRACE")
    print("=" * 60)
    print(f"• Query Type       : {args.type.upper()}")
    if args.type == "trake":
        print(f"• Sub-events ({len(args.events)}) :")
        for i, ev in enumerate(args.events, 1):
            print(f"    E{i}: {ev}")
    else:
        print(f"• Visual Query     : {args.query}")
    if args.ocr:
        print(f"• OCR Filter Text  : {args.ocr}")
    if args.audio:
        print(f"• Audio Transcript : {args.audio}")
    print(f"• Limit Requested  : {args.limit}")
    print(f"• Total Hits Found : {total_results}")
    print("=" * 60 + "\n")


def decompose_into_visual_clauses(text):
    import re
    raw_clauses = re.split(r'[\n\.\;]+', text)
    fillers = [
        r"^đoạn video ghi lại\s*",
        r"^trong clip có cảnh\s*",
        r"^sau đó có cảnh\s*",
        r"^tiếp đó xuất hiện\s*",
        r"^cuối cùng là\s*",
        r"^hình ảnh cho thấy\s*"
    ]
    clauses = []
    for c in raw_clauses:
        c_str = c.strip()
        if not c_str:
            continue
        for f in fillers:
            c_str = re.sub(f, "", c_str, flags=re.IGNORECASE).strip()
        if len(c_str.split()) >= 3:
            clauses.append(c_str)
    return clauses if clauses else [text.strip()]


def main():
    args = parse_args()

    # Dynamic import after setting sys.path
    from retrieval_system import RetrievalSystem

    print(f"[INFO] Initializing RetrievalSystem...")
    system = RetrievalSystem(re_ingest=False)

    results = []

    if args.type in ("kis", "qa"):
        if not args.query and not args.subqueries:
            print("[ERROR] --query or --subqueries is required for 'kis' and 'qa' query types.", file=sys.stderr)
            sys.exit(1)

        if args.subqueries:
            clauses = args.subqueries
        elif args.decompose and ('.' in args.query or '\n' in args.query or ';' in args.query):
            clauses = decompose_into_visual_clauses(args.query)
        else:
            clauses = [args.query]

        if len(clauses) > 1:
            print(f"[INFO] Decomposed into {len(clauses)} atomic visual sub-clauses:")
            for i, cl in enumerate(clauses, 1):
                print(f"       C{i}: '{cl}'")
            print(f"[INFO] Executing multi-clause search with neighborhood window = {args.neighborhood_window} frames...")
            fetch_limit = args.limit * 3
            clause_results = []
            for cl in clauses:
                raw_c = system.semantic_search(
                    query=cl,
                    model_names=args.models,
                    limit=fetch_limit,
                    ocr_query=args.ocr,
                    audio_query=args.audio,
                )
                clause_results.append(raw_c)

            video_frames = {}
            for c_idx, res in enumerate(clause_results):
                for r in res:
                    vid = r.get("video_id")
                    fid = r.get("keyframe_idx")
                    score = r.get("score", 0.0)
                    pts = r.get("pts_time")
                    if vid and fid is not None:
                        if vid not in video_frames:
                            video_frames[vid] = [[] for _ in range(len(clauses))]
                        video_frames[vid][c_idx].append((fid, score, pts))

            scored_candidates = []
            for vid, c_lists in video_frames.items():
                for c_idx in range(len(clauses)):
                    for fid, score, pts in c_lists[c_idx]:
                        fused = score
                        matched_other = 0
                        for other_c in range(len(clauses)):
                            if other_c == c_idx:
                                continue
                            support = [sc for (pf, sc, _) in c_lists[other_c] if abs(pf - fid) <= args.neighborhood_window]
                            if support:
                                fused += 0.8 * max(support)
                                matched_other += 1
                        scored_candidates.append({
                            "video_id": vid,
                            "frame_id": fid,
                            "pts_time": pts,
                            "score": fused,
                            "matched_other": matched_other
                        })

            scored_candidates.sort(key=lambda x: (x["matched_other"], x["score"]), reverse=True)
            selected_by_video = {}
            for item in scored_candidates:
                vid = item["video_id"]
                fid = item["frame_id"]
                if args.dedup_window > 0:
                    past_frames = selected_by_video.get(vid, [])
                    if any(abs(fid - pf) <= args.dedup_window for pf in past_frames):
                        continue
                    selected_by_video.setdefault(vid, []).append(fid)
                results.append(item)
                if len(results) >= args.limit:
                    break
        else:
            print(f"[INFO] Executing {args.mode} search for query: '{clauses[0]}'...")
            fetch_limit = args.limit * 3 if args.dedup_window > 0 else args.limit
            if args.mode == "caption":
                raw_results = system.caption_search(
                    query=clauses[0],
                    limit=fetch_limit,
                    ocr_query=args.ocr,
                    audio_query=args.audio,
                )
            elif args.mode == "both":
                raw_results = system.fused_search(
                    query=clauses[0],
                    model_names=args.models,
                    limit=fetch_limit,
                    ocr_query=args.ocr,
                    audio_query=args.audio,
                )
            else:
                raw_results = system.semantic_search(
                    query=clauses[0],
                    model_names=args.models,
                    limit=fetch_limit,
                    ocr_query=args.ocr,
                    audio_query=args.audio,
                )

            selected_by_video = {}
            for r in raw_results:
                vid = r.get("video_id")
                fid = r.get("keyframe_index") if r.get("keyframe_index") is not None else r.get("keyframe_idx")
                if vid is None or fid is None:
                    continue

                if args.dedup_window > 0:
                    past_frames = selected_by_video.get(vid, [])
                    if any(abs(fid - pf) <= args.dedup_window for pf in past_frames):
                        continue
                    selected_by_video.setdefault(vid, []).append(fid)

                results.append({
                    "video_id": vid,
                    "frame_id": fid,
                    "pts_time": r.get("pts_time"),
                    "score": r.get("score", 0.0),
                })
                if len(results) >= args.limit:
                    break

    elif args.type == "trake":
        if not args.events or len(args.events) < 2:
            print("[ERROR] --events requires at least 2 sub-events for 'trake'.", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO] Executing TRAKE temporal search for {len(args.events)} events...")
        event_queries = [{"text": ev} for ev in args.events]
        raw_results = system.temporal_search(
            queries=event_queries,
            model_names=args.models,
            group_by_video=True,
            limit=args.limit,
        )

        for r in raw_results:
            results.append({
                "video_id": r.get("video_id"),
                "frames": r.get("path", []),
                "scores": r.get("scores", []),
                "total_score": r.get("total_score", 0.0),
            })

    if args.log_trace:
        print_search_trace(args, len(results))

    # Print top 10 preview
    print("Top Candidate Preview:")
    print("-" * 50)
    for i, item in enumerate(results[:10], 1):
        if args.type in ("kis", "qa"):
            print(f"#{i:02d} | Video: {item['video_id']} | Frame: {item['frame_id']:05d} | Score: {item['score']:.4f}")
        else:
            frames_str = ", ".join(str(f) for f in item.get("frames", []))
            print(f"#{i:02d} | Video: {item['video_id']} | Frames: [{frames_str}] | Score: {item['total_score']:.4f}")
    print("-" * 50)

    # Save to output file if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix.lower() == ".json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] Saved {len(results)} records to JSON: {out_path}")
        else:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if args.type == "kis":
                    for item in results:
                        writer.writerow([item["video_id"], item["frame_id"]])
                elif args.type == "qa":
                    for item in results:
                        # Placeholder empty answer for QA to be filled via visual inspection
                        writer.writerow([item["video_id"], item["frame_id"], ""])
                elif args.type == "trake":
                    for item in results:
                        writer.writerow([item["video_id"]] + list(item.get("frames", [])))
            print(f"[SUCCESS] Saved {len(results)} records to CSV: {out_path}")


if __name__ == "__main__":
    main()
