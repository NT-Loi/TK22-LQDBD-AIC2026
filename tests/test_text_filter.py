import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from retrieval_system import RetrievalSystem
from config import EMBEDDING_WEIGHTS

class TestTextFilter(unittest.TestCase):
    def setUp(self):
        self.system = RetrievalSystem.__new__(RetrievalSystem)
        self.system.text_encoders = {
            "SigLIP2": MagicMock(return_value=np.zeros((1, 1536))),
            "Qwen3_VL_Embedding": MagicMock(return_value=np.zeros((1, 2048))),
        }
        self.system.encode_query = MagicMock(return_value={
            "SigLIP2": np.zeros((1, 1536)),
        })
        self.system.qdrant_client = MagicMock()
        # Mock shot data: Video 1 has shots [0, 200] and [201, 500]
        self.system.shots_data = {
            "V1": [(0, 200), (201, 500)],
            "V2": [(0, 300)]
        }
        self.system.video_metadata = {}

    def test_text_filter_and_logic_non_temporal(self):
        """
        Verify that multiple text query filters use commutative AND logic.
        The order of queries in text_filters should not affect the intersected videos.
        """
        # Query A hits V1 and V2
        # Query B hits V1 and V3
        def mock_query_points(collection_name, query, using, limit, search_params, with_payload):
            mock_res = MagicMock()
            if self.current_query == "filter_a":
                hit1 = MagicMock(id="p1", score=0.9, payload={"video_id": "V1", "keyframe_idx": 50})
                hit2 = MagicMock(id="p2", score=0.8, payload={"video_id": "V2", "keyframe_idx": 100})
                mock_res.points = [hit1, hit2]
            elif self.current_query == "filter_b":
                hit1 = MagicMock(id="p3", score=0.85, payload={"video_id": "V1", "keyframe_idx": 60})
                hit3 = MagicMock(id="p4", score=0.75, payload={"video_id": "V3", "keyframe_idx": 150})
                mock_res.points = [hit1, hit3]
            else:
                mock_res.points = []
            return mock_res

        self.system.qdrant_client.query_points.side_effect = mock_query_points

        def mock_encode(text):
            self.current_query = text
            return np.zeros((1, 1536))

        self.system.text_encoders["SigLIP2"].side_effect = mock_encode

        # Test Order 1: [A, B]
        filters_order1 = [
            {"text": "filter_a", "level": "frame"},
            {"text": "filter_b", "level": "video"}
        ]
        res1 = self.system._get_text_filter_ranges(filters_order1)
        self.assertIn("common_vids", res1)
        self.assertEqual(res1["common_vids"], {"V1"})

        # Test Order 2: [B, A] (Order reversed -> Non-temporal commutative result)
        filters_order2 = [
            {"text": "filter_b", "level": "video"},
            {"text": "filter_a", "level": "frame"}
        ]
        res2 = self.system._get_text_filter_ranges(filters_order2)
        self.assertIn("common_vids", res2)
        self.assertEqual(res2["common_vids"], {"V1"})

    def test_frame_level_vs_video_level(self):
        """
        Verify frame-level precision vs video-level broad matching.
        """
        filter_data = {
            "common_vids": {"V1"},
            "query_items": [
                {
                    "text": "red car",
                    "level": "frame",
                    "vid_map": {
                        "V1": {
                            50: {"score": 0.9, "text": "red car"}
                        }
                    }
                },
                {
                    "text": "outdoor street",
                    "level": "video",
                    "vid_map": {
                        "V1": {
                            400: {"score": 0.8, "text": "outdoor street"}
                        }
                    }
                }
            ]
        }

        # Frame 50 is exact match for frame filter and V1 satisfies video filter -> PASS
        score_50 = self.system._frame_in_text_filter_ranges("V1", 50, filter_data)
        self.assertGreater(score_50, 0.0)

        # Frame 100 is within proximity window (<= 150 frames) and same shot [0, 200] as 50 -> PASS
        score_100 = self.system._frame_in_text_filter_ranges("V1", 100, filter_data)
        self.assertGreater(score_100, 0.0)

        # Frame 450 is in shot [201, 500] and distance to 50 is 400 (> 150) -> FAIL frame-level filter
        score_450 = self.system._frame_in_text_filter_ranges("V1", 450, filter_data)
        self.assertEqual(score_450, 0.0)

        # Video V2 is not in common_vids -> FAIL
        score_v2 = self.system._frame_in_text_filter_ranges("V2", 50, filter_data)
        self.assertEqual(score_v2, 0.0)

    def test_semantic_search_filtered_by_text_query(self):
        """
        Verify that semantic_search eliminates candidates that fail text_filters.
        """
        hit1 = MagicMock(id="p1", score=10.0, payload={"video_id": "V1", "keyframe_idx": 50})
        hit2 = MagicMock(id="p2", score=9.0, payload={"video_id": "V1", "keyframe_idx": 450})

        filter_data = {
            "common_vids": {"V1"},
            "query_items": [
                {
                    "text": "white dog",
                    "level": "frame",
                    "vid_map": {
                        "V1": {
                            50: {"score": 1.0, "text": "white dog"}
                        }
                    }
                }
            ]
        }

        def mock_query_points(collection_name, query, using, limit, search_params, with_payload):
            mock_res = MagicMock()
            mock_res.points = [hit1, hit2]
            return mock_res

        self.system.qdrant_client.query_points.side_effect = mock_query_points

        with patch.dict(EMBEDDING_WEIGHTS, {"SigLIP2": 1.0}):
            results = self.system.semantic_search(
                "main event query",
                model_names=["SigLIP2"],
                top_k=10,
                limit=10,
                text_filters=filter_data
            )

        # Only frame 50 should remain; frame 450 must be filtered out
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["keyframe_index"], 50)

    def test_filter_search_with_text_filter_only(self):
        """
        Verify that filter_search works when only text_filters are provided (no event query).
        """
        filter_data = {
            "common_vids": {"V1"},
            "query_items": [
                {
                    "text": "traffic light",
                    "level": "frame",
                    "vid_map": {
                        "V1": {
                            75: {"score": 0.88, "text": "traffic light"}
                        }
                    }
                }
            ]
        }

        results = self.system.filter_search(
            text_filters=filter_data,
            limit=10
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["video_id"], "V1")
        self.assertEqual(results[0]["keyframe_index"], 75)
        self.assertAlmostEqual(results[0]["score"], 0.88, places=2)

if __name__ == "__main__":
    unittest.main()
