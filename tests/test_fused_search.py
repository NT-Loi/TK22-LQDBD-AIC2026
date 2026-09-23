import unittest
from unittest.mock import MagicMock
from retrieval_system import RetrievalSystem

class TestFusedSearch(unittest.TestCase):
    def test_fused_score_preservation_and_synergy(self):
        system = RetrievalSystem.__new__(RetrievalSystem)
        system.shots_data = {}

        # Mock semantic_search (keyframe search)
        system.semantic_search = MagicMock(return_value=[
            {"video_id": "V1", "keyframe_index": 10, "score": 0.8},
            {"video_id": "V1", "keyframe_index": 20, "score": 0.4},
        ])

        # Mock caption_search (shot caption search)
        system.caption_search = MagicMock(return_value=[
            {
                "video_id": "V1",
                "shot_idx": 1,
                "shot_start_frame": 18,
                "shot_end_frame": 22,
                "representative_frame": 20,
                "score": 0.9,
                "caption_text": "cap 1",
                "best_aspect": "chu_the_hanh_dong",
                "best_aspect_caption": "cap 1",
            },
            {
                "video_id": "V1",
                "shot_idx": 2,
                "shot_start_frame": 28,
                "shot_end_frame": 32,
                "representative_frame": 30,
                "score": 0.3,
                "caption_text": "cap 2",
                "best_aspect": "tong_the_canh_quay",
                "best_aspect_caption": "cap 2",
            },
        ])
        system._get_keyframes_in_shot = MagicMock(side_effect=lambda vid, s, e: [20] if s == 18 else [30])

        results = system.fused_search("test query", limit=10)
        
        res_map = {(r["video_id"], r["keyframe_index"]): r for r in results}
        self.assertIn(("V1", 10), res_map)
        self.assertIn(("V1", 20), res_map)
        self.assertIn(("V1", 30), res_map)

        # (V1, 10) is KF-only: raw score 0.8 is max in KF -> normalized 1.0
        # MUST remain 1.0 (NOT halved to 0.5)
        r10 = res_map[("V1", 10)]
        self.assertEqual(r10["keyframe_score"], 1.0)
        self.assertEqual(r10["caption_score"], 0.0)
        self.assertEqual(r10["score"], 1.0)

        # (V1, 20) is in BOTH KF and Caption:
        # kf_norm = (0.4 - 0.4)/(0.8 - 0.4) = 0.0
        # cap_norm = (0.9 - 0.3)/(0.9 - 0.3) = 1.0
        # base_score = max((0.5*0 + 0.5*1.0), max(0, 1.0)) = 1.0
        # synergy = 0.15 * min(0, 1.0) = 0.0
        # fused = 1.0
        r20 = res_map[("V1", 20)]
        self.assertEqual(r20["keyframe_score"], 0.0)
        self.assertEqual(r20["caption_score"], 1.0)
        self.assertEqual(r20["score"], 1.0)

        # Now test with both modalities having positive normalized score
        # so we verify synergy bonus
        system.semantic_search = MagicMock(return_value=[
            {"video_id": "V1", "keyframe_index": 100, "score": 0.8},
            {"video_id": "V1", "keyframe_index": 101, "score": 0.0},
        ])
        system.caption_search = MagicMock(return_value=[
            {
                "video_id": "V1",
                "shot_idx": 1,
                "shot_start_frame": 100,
                "shot_end_frame": 100,
                "representative_frame": 100,
                "score": 0.8,
            },
            {
                "video_id": "V1",
                "shot_idx": 2,
                "shot_start_frame": 102,
                "shot_end_frame": 102,
                "representative_frame": 102,
                "score": 0.0,
            }
        ])
        system._get_keyframes_in_shot = MagicMock(side_effect=lambda vid, s, e: [s])

        results2 = system.fused_search("test query 2", limit=10)
        res_map2 = {(r["video_id"], r["keyframe_index"]): r for r in results2}

        # (V1, 100) has kf_norm = 1.0, cap_norm = 1.0
        # base_score = 1.0, synergy = 0.15*1.0 = 0.15 -> min(1.0, 1.15) = 1.0
        self.assertEqual(res_map2[("V1", 100)]["score"], 1.0)

        # Test partial scores to see synergy clearly:
        # KF: (V1, 200) raw 0.75 in [0.5, 1.0] -> norm 0.50
        # Cap: (V1, 200) raw 0.70 in [0.4, 1.0] -> norm 0.50
        # Single match (V1, 201): KF raw 0.75 -> norm 0.50
        system.semantic_search = MagicMock(return_value=[
            {"video_id": "V1", "keyframe_index": 200, "score": 0.75},
            {"video_id": "V1", "keyframe_index": 201, "score": 0.75},
            {"video_id": "V1", "keyframe_index": 202, "score": 0.50},
            {"video_id": "V1", "keyframe_index": 203, "score": 1.00},
        ])
        system.caption_search = MagicMock(return_value=[
            {
                "video_id": "V1",
                "shot_idx": 1,
                "shot_start_frame": 200,
                "shot_end_frame": 200,
                "representative_frame": 200,
                "score": 0.70,
            },
            {
                "video_id": "V1",
                "shot_idx": 2,
                "shot_start_frame": 202,
                "shot_end_frame": 202,
                "representative_frame": 202,
                "score": 0.40,
            },
            {
                "video_id": "V1",
                "shot_idx": 3,
                "shot_start_frame": 203,
                "shot_end_frame": 203,
                "representative_frame": 203,
                "score": 1.00,
            },
        ])
        results3 = system.fused_search("test query 3", limit=10)
        res_map3 = {(r["video_id"], r["keyframe_index"]): r for r in results3}

        # (V1, 201) is KF-only with norm 0.50. Fused MUST be exactly 0.50 (NOT 0.25!)
        self.assertAlmostEqual(res_map3[("V1", 201)]["score"], 0.50, places=4)

        # (V1, 200) is dual match with kf_norm=0.50 and cap_norm=0.50.
        # base_score = 0.50, synergy = 0.15 * 0.50 = 0.075 -> fused = 0.575
        # Dual match MUST rank strictly higher than single match (0.575 > 0.50)!
        self.assertAlmostEqual(res_map3[("V1", 200)]["score"], 0.575, places=4)
        self.assertGreater(res_map3[("V1", 200)]["score"], res_map3[("V1", 201)]["score"])

if __name__ == "__main__":
    unittest.main()
