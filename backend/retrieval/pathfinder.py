import logging

logger = logging.getLogger(__name__)

def find_temporal_sequences(query_results: list, queries: list, group_by_shot: bool, max_frame_gap: int, limit: int = 100) -> list:
    """
    Finds valid chronological event combinations across search query results.
    Guarantees subsequent events happen strictly after previous ones and within the frame gap.
    """
    video_grouped = {}
    for q_idx, res_list in enumerate(query_results):
        for item in res_list:
            vid = item["video_id"]
            if vid not in video_grouped:
                video_grouped[vid] = [[] for _ in range(len(queries))]
            video_grouped[vid][q_idx].append(item)
            
    all_sequences = []
    
    for vid, vid_group in video_grouped.items():
        if any(len(q_res) == 0 for q_res in vid_group):
            continue
        
        for i in range(len(queries)):
            if group_by_shot:
                vid_group[i].sort(key=lambda x: x["shot_start_frame"])
            else:
                vid_group[i].sort(key=lambda x: x["keyframe_index"])
        
        def find_paths(current_q_idx, current_path):
            if current_q_idx == len(queries):
                all_sequences.append(list(current_path))
                return
                
            for candidate in vid_group[current_q_idx]:
                if len(current_path) == 0:
                    current_path.append(candidate)
                    find_paths(current_q_idx + 1, current_path)
                    current_path.pop()
                else:
                    prev = current_path[-1]
                    valid = False
                    
                    if not group_by_shot:
                        gap = candidate["keyframe_index"] - prev["keyframe_index"]
                        if 0 < gap <= max_frame_gap:
                            valid = True
                    else:
                        if candidate["shot_start_frame"] > prev["shot_end_frame"]:
                            gap = candidate["shot_start_frame"] - prev["shot_end_frame"]
                            if gap <= max_frame_gap:
                                valid = True
                                
                    if valid:
                        current_path.append(candidate)
                        find_paths(current_q_idx + 1, current_path)
                        current_path.pop()

        find_paths(0, [])
        
    final_results = []
    for seq in all_sequences:
        avg_score = sum(item["score"] for item in seq) / len(seq)
        
        if group_by_shot:
            flat_frames = []
            display_frames = []
            for cand in seq:
                sorted_items = sorted(cand["frames"], key=lambda x: x["keyframe_index"])
                flat_frames.extend(sorted_items)
                best_frame = max(cand["frames"], key=lambda x: x["score"])
                display_frames.append(best_frame)
            anchor = display_frames[0]
        else:
            flat_frames = seq
            display_frames = seq
            anchor = seq[0]
        
        final_results.append({
            "video_id": anchor["video_id"],
            "sequence_score": avg_score,
            "frames": flat_frames,
            "display_frames": display_frames,
            "keyframe_index": anchor["keyframe_index"],
            "shot_start_frame": anchor.get("shot_start_frame"),
            "shot_end_frame": anchor.get("shot_end_frame"),
            "score": avg_score, 
        })
        
    final_results = sorted(final_results, key=lambda x: x["sequence_score"], reverse=True)
    return final_results[:limit]
