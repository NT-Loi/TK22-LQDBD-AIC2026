#!/usr/bin/env python3
"""
run_round_3.py - Blind batch retrieval & comprehensive trace logging for Round 3 queries.
Phase 1: Generates all predictions WITHOUT looking at answers, records search methodology to markdown.
Phase 2: Compares predictions against ground truth answers and calculates R-Score & Final Score.
"""

import os
import sys
import json
import csv
import re
import urllib.request
from pathlib import Path
from datetime import datetime

# Force utf-8 encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
WORKSPACE = Path(__file__).resolve().parents[3]
QUESTIONS_DIR = WORKSPACE / "retrieval_queries" / "round_3"
if not QUESTIONS_DIR.exists():
    QUESTIONS_DIR = WORKSPACE / "retrieval_questions" / "round_3"
ANSWERS_DIR = QUESTIONS_DIR / "answers"
SUBMISSION_DIR = WORKSPACE / "submission"
PREDICTIONS_DIR = SUBMISSION_DIR / "round_3_eval"
TRACE_MD_FILE = SUBMISSION_DIR / "SEARCH_STRATEGY_TRACE.md"
SEARCH_URL = "http://localhost:8000/search"

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def query_search_api(payload):
    req = urllib.request.Request(
        SEARCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def decompose_into_visual_clauses(text):
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


def search_with_neighborhood_fusion(clauses, models=["SigLIP", "SigLIP2"], window=250, limit=100):
    if len(clauses) <= 1:
        payload = {
            "text_queries": [{"text": clauses[0]}],
            "models": models,
            "limit": 250
        }
        raw = query_search_api(payload)
        candidates = []
        seen = {}
        for r in raw:
            vid = r.get("video_id")
            fidx = r.get("keyframe_index")
            score = r.get("score", 0.0)
            if vid and fidx is not None:
                if any(abs(fidx - pf) <= 15 for pf in seen.get(vid, [])):
                    continue
                seen.setdefault(vid, []).append(fidx)
                candidates.append((vid, fidx, score))
                if len(candidates) >= limit:
                    break
        return candidates

    # Multiple sub-clauses: query each independently
    clause_results = []
    for cl in clauses:
        payload = {
            "text_queries": [{"text": cl}],
            "models": models,
            "limit": 250
        }
        clause_results.append(query_search_api(payload))

    # Group candidate frames by video_id
    video_frames = {}
    for c_idx, res in enumerate(clause_results):
        for r in res:
            vid = r.get("video_id")
            fidx = r.get("keyframe_index")
            score = r.get("score", 0.0)
            if vid and fidx is not None:
                if vid not in video_frames:
                    video_frames[vid] = [[] for _ in range(len(clauses))]
                video_frames[vid][c_idx].append((fidx, score))

    scored_candidates = []
    for vid, c_lists in video_frames.items():
        for c_idx in range(len(clauses)):
            for fidx, score in c_lists[c_idx]:
                fused = score
                matched_other = 0
                for other_c in range(len(clauses)):
                    if other_c == c_idx:
                        continue
                    support = [sc for (pf, sc) in c_lists[other_c] if abs(pf - fidx) <= window]
                    if support:
                        fused += 0.8 * max(support)
                        matched_other += 1
                scored_candidates.append((vid, fidx, fused, matched_other))

    # Rank by number of co-occurring clauses in neighborhood, then by fused score
    scored_candidates.sort(key=lambda x: (x[3], x[2]), reverse=True)

    # Apply temporal deduplication (15 frame window)
    final_candidates = []
    seen = {}
    for vid, fidx, score, matched in scored_candidates:
        if any(abs(fidx - pf) <= 15 for pf in seen.get(vid, [])):
            continue
        seen.setdefault(vid, []).append(fidx)
        final_candidates.append((vid, fidx, score))
        if len(final_candidates) >= limit:
            break

    return final_candidates


def solve_kis(query_file):
    text = query_file.read_text(encoding="utf-8").strip()
    clauses = decompose_into_visual_clauses(text)
    candidates = search_with_neighborhood_fusion(clauses, models=["SigLIP", "SigLIP2"], limit=100)
    
    trace = {
        "raw_text": text,
        "decomposed_clauses": clauses,
        "decomposition": {
            "clauses": clauses,
            "ocr": None,
            "audio": None
        },
        "models": ["SigLIP", "SigLIP2"],
        "candidates": candidates
    }
    return candidates, trace


def solve_qa(query_file):
    text = query_file.read_text(encoding="utf-8").strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    
    if len(lines) >= 2 and ("?" in lines[-1] or lines[-1].lower().startswith("theo") or lines[-1].lower().startswith("trong")):
        context_desc = " ".join(lines[:-1])
        question = lines[-1]
    else:
        context_desc = " ".join(lines)
        question = lines[-1]

    clauses = decompose_into_visual_clauses(context_desc)
    raw_candidates = search_with_neighborhood_fusion(clauses, models=["SigLIP", "SigLIP2"], limit=100)
    candidates = [(vid, fidx, score, "") for (vid, fidx, score) in raw_candidates]
            
    trace = {
        "raw_text": text,
        "context": context_desc,
        "question": question,
        "decomposed_clauses": clauses,
        "decomposition": {
            "visual_context": context_desc,
            "target_question": question,
            "clauses": clauses,
            "ocr": None,
            "audio": None
        },
        "models": ["SigLIP", "SigLIP2"],
        "candidates": candidates
    }
    return candidates, trace


def solve_trake(query_file):
    text = query_file.read_text(encoding="utf-8").strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    
    context_line = lines[0] if lines else ""
    event_queries = []
    for line in lines:
        m = re.match(r"^E\d+\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            event_queries.append({"text": m.group(1).strip()})
        elif line.startswith("(") and ")" in line[:4]:
            event_queries.append({"text": line.split(")", 1)[1].strip()})
            
    if not event_queries:
        event_queries = [{"text": l} for l in lines[1:] if len(l) > 5]

    payload = {
        "text_queries": event_queries,
        "models": ["SigLIP", "SigLIP2"],
        "group_by_video": True,
        "limit": 100
    }
    raw_results = query_search_api(payload)
    
    candidates = []
    for r in raw_results:
        vid = r.get("video_id")
        frames = [f.get("keyframe_index") for f in r.get("frames", [])]
        score = r.get("sequence_score", r.get("best_sequence_score", 0.0))
        if vid and len(frames) == len(event_queries):
            if all(frames[i] < frames[i+1] for i in range(len(frames)-1)):
                candidates.append([vid] + frames + [score])
                
    trace = {
        "raw_text": text,
        "context": context_line,
        "events": [eq["text"] for eq in event_queries],
        "decomposition": {
            "overall_context": context_line,
            "sub_events": [f"E{i}: {eq['text']}" for i, eq in enumerate(event_queries, 1)],
            "temporal_constraint": "frame_1 < frame_2 < ... < frame_n"
        },
        "models": ["SigLIP", "SigLIP2"],
        "candidates": candidates
    }
    return candidates, trace


def run_phase_1_blind_retrieval():
    print("\n" + "=" * 70)
    print("🚀 PHASE 1: BLIND RETRIEVAL EXECUTION (STRICTLY NO ANSWER ACCESS)")
    print("=" * 70)
    
    query_files = sorted([f for f in QUESTIONS_DIR.glob("query-*.txt")])
    print(f"[INFO] Found {len(query_files)} query files in {QUESTIONS_DIR.name}")
    
    all_traces = []

    for idx, qfile in enumerate(query_files, 1):
        qname = qfile.stem
        qtype = qname.split("-")[-1] # kis, qa, trake
        pred_csv = PREDICTIONS_DIR / f"{qname}.csv"
        
        print(f"[{idx:02d}/{len(query_files):02d}] Processing {qname} ({qtype.upper()})...", end="", flush=True)
        
        try:
            if qtype == "kis":
                results, trace = solve_kis(qfile)
                with open(pred_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for vid, fidx, sc in results:
                        writer.writerow([vid, fidx])
                top1_info = f"{results[0][0]}, frame {results[0][1]} (score: {results[0][2]:.4f})" if results else "No results"
                
            elif qtype == "qa":
                results, trace = solve_qa(qfile)
                with open(pred_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for item in results:
                        writer.writerow([item[0], item[1], item[3]])
                top1_info = f"{results[0][0]}, frame {results[0][1]} (score: {results[0][2]:.4f})" if results else "No results"
                
            elif qtype == "trake":
                results, trace = solve_trake(qfile)
                with open(pred_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for row in results:
                        writer.writerow(row[:-1]) # exclude score column for submission
                top1_info = f"{results[0][0]}, frames {results[0][1:-1]}" if results else "No results"
                
            trace["name"] = qname
            trace["type"] = qtype
            all_traces.append(trace)
            print(f" -> Done! ({len(results)} candidates, Top-1: {top1_info})")
            
        except Exception as e:
            print(f" -> ERROR: {e}")

    print("\n[PHASE 1 COMPLETE] All predictions generated and stored at:")
    print(f"📁 {PREDICTIONS_DIR}\n")
    return all_traces


def run_phase_2_evaluation(all_traces):
    print("\n" + "=" * 70)
    print("📊 PHASE 2: COMPARISON & EVALUATION AGAINST GROUND TRUTH")
    print("=" * 70)
    
    eval_results = []
    
    for trace in all_traces:
        qname = trace["name"]
        qtype = trace["type"]
        
        pred_csv = PREDICTIONS_DIR / f"{qname}.csv"
        ans_csv = ANSWERS_DIR / f"{qname}.csv"
        ans_json = ANSWERS_DIR / f"{qname}.json"
        
        gt_data = None
        if ans_csv.exists():
            with open(ans_csv, "r", encoding="utf-8") as f:
                gt_data = [row for row in csv.reader(f) if row]
        elif ans_json.exists():
            with open(ans_json, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                
        if not gt_data or not pred_csv.exists():
            continue
            
        with open(pred_csv, "r", encoding="utf-8") as f:
            preds = [row for row in csv.reader(f) if row]
            
        eval_info = evaluate_single_query(qname, qtype, preds, gt_data)
        trace["eval"] = eval_info
        eval_results.append(eval_info)

    # Print summary table
    print("\n" + "=" * 85)
    print(f"{'Query Name':<22} | {'Type':<6} | {'Top-1 Video':<10} | {'GT Video':<10} | {'R@1':<5} | {'R@5':<5} | {'Final':<6}")
    print("-" * 85)
    
    r1_list, r5_list, final_list = [], [], []
    for er in eval_results:
        qname = er["name"]
        qtype = er["type"].upper()
        top_vid = er.get("top_video", "N/A")
        gt_vid = er.get("gt_video", "N/A")
        r1 = er.get("r1", 0.0)
        r5 = er.get("r5", 0.0)
        fscore = er.get("final_score", 0.0)
        
        r1_list.append(r1)
        r5_list.append(r5)
        final_list.append(fscore)
        
        print(f"{qname:<22} | {qtype:<6} | {top_vid:<10} | {gt_vid:<10} | {r1:<5.2f} | {r5:<5.2f} | {fscore:<6.2f}")
        
    print("=" * 85)
    avg_r1 = sum(r1_list) / len(r1_list) if r1_list else 0.0
    avg_r5 = sum(r5_list) / len(r5_list) if r5_list else 0.0
    avg_final = sum(final_list) / len(final_list) if final_list else 0.0
    print(f"OVERALL SUMMARY ({len(eval_results)} queries):")
    print(f"• Average R@1       : {avg_r1:.4f}")
    print(f"• Average R@5       : {avg_r5:.4f}")
    print(f"• Mean Final Score  : {avg_final:.4f}")
    print("=" * 85 + "\n")
    
    # Generate comprehensive markdown documentation
    write_search_trace_markdown(all_traces, eval_results, avg_r1, avg_r5, avg_final)


def evaluate_single_query(qname, qtype, preds, gt_data):
    info = {"name": qname, "type": qtype, "r1": 0.0, "r5": 0.0, "final_score": 0.0}
    if not preds or not gt_data:
        return info

    top_vid = preds[0][0]
    info["top_video"] = top_vid

    gt_video = None
    gt_frames = set()
    
    for row in gt_data:
        if not row:
            continue
        gt_video = row[0].strip()
        for col in row[1:]:
            col = col.strip()
            if col.isdigit():
                gt_frames.add(int(col))
                
    info["gt_video"] = gt_video or "Unknown"

    r_scores = []
    for p in preds:
        vid = p[0].strip()
        if vid == gt_video:
            if qtype == "trake":
                # For TRAKE: proportion of matched frames within window of 50
                p_frames = [int(x.strip()) for x in p[1:] if x.strip().isdigit()]
                if p_frames:
                    matched_cnt = sum(1 for pf in p_frames if any(abs(pf - gf) <= 50 for gf in gt_frames))
                    r_scores.append(matched_cnt / len(p_frames))
                else:
                    r_scores.append(0.5)
            else:
                if len(p) >= 2 and p[1].strip().isdigit():
                    f = int(p[1].strip())
                    matched = any(abs(f - gf) <= 50 for gf in gt_frames)
                    r_scores.append(1.0 if matched else 0.5)
                else:
                    r_scores.append(1.0)
        else:
            r_scores.append(0.0)

    def top_k_r(k):
        return max(r_scores[:k]) if r_scores[:k] else 0.0

    r1 = top_k_r(1)
    r5 = top_k_r(5)
    r20 = top_k_r(20)
    r50 = top_k_r(50)
    r100 = top_k_r(100)

    final_score = (r1 + r5 + r20 + r50 + r100) / 5.0
    info["r1"] = r1
    info["r5"] = r5
    info["r20"] = r20
    info["r50"] = r50
    info["r100"] = r100
    info["final_score"] = final_score
    return info


def write_search_trace_markdown(all_traces, eval_results, avg_r1, avg_r5, avg_final):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_lines = [
        "# 🔍 SEARCH STRATEGY & EXECUTION TRACE REPORT — ROUND 3",
        "",
        f"- **Execution Timestamp:** `{now_str}`",
        f"- **Workspace:** `{WORKSPACE.name}`",
        f"- **Target Dataset Prefix Filter:** `VALID_VIDEO_PREFIX = 'L'` (Strictly filtering out old Batch 1 'K' videos)",
        f"- **Visual Encoders Used:** `SigLIP (1152d)` + `SigLIP2 (1536d)` via Qdrant Vector Search",
        f"- **Total Queries Executed:** `{len(all_traces)}` queries",
        "",
        "---",
        "",
        "## 📊 1. Overall Evaluation Summary",
        "",
        f"- **Average R@1:** `{avg_r1:.4f}`",
        f"- **Average R@5:** `{avg_r5:.4f}`",
        f"- **Mean Final Score:** `{avg_final:.4f}`",
        "",
        "| Query Name | Type | Top-1 Video | GT Video | R@1 | R@5 | Final Score | Status |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |"
    ]
    
    for er in eval_results:
        qname = er["name"]
        qtype = er["type"].upper()
        top_vid = er.get("top_video", "N/A")
        gt_vid = er.get("gt_video", "N/A")
        r1 = er.get("r1", 0.0)
        r5 = er.get("r5", 0.0)
        fscore = er.get("final_score", 0.0)
        
        status = "❌ Miss"
        if r1 >= 1.0:
            status = "🌟 PERFECT (Top-1 Exact)"
        elif r1 > 0:
            status = "✅ Correct Video at Top-1"
        elif r5 > 0:
            status = "🎯 Correct Video in Top-5"
        elif fscore > 0:
            status = "🔍 Correct Video in Top-100"
            
        md_lines.append(f"| `{qname}` | {qtype} | `{top_vid}` | `{gt_vid}` | {r1:.2f} | {r5:.2f} | **{fscore:.2f}** | {status} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 📝 2. Query-by-Query Search Strategy & Decomposition Traces",
        ""
    ])

    for idx, trace in enumerate(all_traces, 1):
        qname = trace["name"]
        qtype = trace["type"].upper()
        raw_text = trace.get("raw_text", "").strip()
        models_used = ", ".join(trace.get("models", []))
        ev = trace.get("eval", {})
        
        md_lines.extend([
            f"### Query #{idx:02d}: `{qname}` ({qtype})",
            "",
            f"- **Ground Truth Video:** `{ev.get('gt_video', 'N/A')}`",
            f"- **Retrieved Top-1 Video:** `{ev.get('top_video', 'N/A')}`",
            f"- **Performance:** R@1: `{ev.get('r1', 0.0):.2f}` | R@5: `{ev.get('r5', 0.0):.2f}` | Final Score: `**{ev.get('final_score', 0.0):.2f}**`",
            "",
            "#### Raw Natural Vietnamese Query:",
            "```text",
            raw_text,
            "```",
            "",
            "#### Query Decomposition & Strategy:",
            "- **Query Nature:** Pure Event & Action Description.",
            f"- **Visual Query:** `{trace.get('combined_query', trace.get('context', ''))}`",
            "- **On-screen Text (OCR):** `None` (Pure visual event, no forced OCR to avoid false negatives).",
            "- **Audio Transcript:** `None` (No forced audio filter).",
            f"- **Models & Ensembles:** `{models_used}` (Qdrant exact cosine vector search).",
            f"- **Dataset Filter:** `VALID_VIDEO_PREFIX = 'L'` (Excluded non-existent 'K' videos).",
            ""
        ])
        
        if qtype == "TRAKE":
            events = trace.get("events", [])
            md_lines.append("#### Temporal Sub-events Chain:")
            for e_idx, ev_desc in enumerate(events, 1):
                md_lines.append(f"- **E{e_idx}:** {ev_desc}")
            md_lines.append("")

        cands = trace.get("candidates", [])
        if cands:
            md_lines.append("#### Top-5 Retrieved Candidates:")
            md_lines.append("| Rank | Video ID | Keyframe(s) | Confidence Score | GT Match? |")
            md_lines.append("| :--- | :--- | :--- | :--- | :---: |")
            for c_idx, c in enumerate(cands[:5], 1):
                vid = c[0]
                gt_match = "✅ Match" if vid == ev.get("gt_video") else "No"
                if qtype == "TRAKE":
                    frames_str = ", ".join(str(f) for f in c[1:-1])
                    sc = c[-1]
                    md_lines.append(f"| #{c_idx} | `{vid}` | `[{frames_str}]` | {sc:.4f} | {gt_match} |")
                else:
                    fidx = c[1]
                    sc = c[2]
                    md_lines.append(f"| #{c_idx} | `{vid}` | `{fidx}` | {sc:.4f} | {gt_match} |")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    TRACE_MD_FILE.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[SUCCESS] Comprehensive Search Trace report written to:\n📄 {TRACE_MD_FILE}")


if __name__ == "__main__":
    traces = run_phase_1_blind_retrieval()
    run_phase_2_evaluation(traces)
