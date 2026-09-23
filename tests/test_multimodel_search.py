import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from retrieval_system import RetrievalSystem
from config import EMBEDDING_WEIGHTS

class TestMultiModelSearch(unittest.TestCase):
    def test_score_normalized_by_participating_models_only(self):
        system = RetrievalSystem.__new__(RetrievalSystem)
        system.text_encoders = {
            "SigLIP2": MagicMock(return_value=np.zeros((1, 1536))),
            "Qwen3_VL_Embedding": MagicMock(return_value=np.zeros((1, 2048))),
        }
        system.encode_query = MagicMock(return_value={
            "SigLIP2": np.zeros((1, 1536)),
            "Qwen3_VL_Embedding": np.zeros((1, 2048)),
        })
        system.qdrant_client = MagicMock()
        system.shots_data = {}

        # Mock point hits
        # Hit 1: in both models
        # Hit 2: only in SigLIP2
        # Hit 3: only in Qwen3_VL_Embedding
        hit1_a = MagicMock(id="p1", score=10.0, payload={"video_id": "V1", "keyframe_idx": 100})
        hit2_a = MagicMock(id="p2", score=5.0, payload={"video_id": "V1", "keyframe_idx": 200})

        hit1_b = MagicMock(id="p1", score=20.0, payload={"video_id": "V1", "keyframe_idx": 100})
        hit3_b = MagicMock(id="p3", score=10.0, payload={"video_id": "V1", "keyframe_idx": 300})

        def mock_query_points(collection_name, query, using, limit, search_params, with_payload):
            mock_res = MagicMock()
            if using == "SigLIP2":
                mock_res.points = [hit1_a, hit2_a]
            elif using == "Qwen3_VL_Embedding":
                mock_res.points = [hit1_b, hit3_b]
            else:
                mock_res.points = []
            return mock_res

        system.qdrant_client.query_points.side_effect = mock_query_points

        with patch.dict(EMBEDDING_WEIGHTS, {"SigLIP2": 1.0, "Qwen3_VL_Embedding": 1.0, "CLIP_H14": 1.0, "FG_CLIP2": 1.0}):
            results = system.semantic_search(
                "test query",
                model_names=["SigLIP2", "Qwen3_VL_Embedding"],
                top_k=10,
                limit=10
            )

        res_by_kf = {r["keyframe_index"]: r["score"] for r in results}
        self.assertIn(100, res_by_kf)
        self.assertIn(200, res_by_kf)
        self.assertIn(300, res_by_kf)

        # In SigLIP2: scores are 10.0 (max) and 5.0 (min)
        # -> p1 (kf 100) norm = 1.0
        # -> p2 (kf 200) norm = 0.0
        # In Qwen3_VL_Embedding: scores are 20.0 (max) and 10.0 (min)
        # -> p1 (kf 100) norm = 1.0
        # -> p3 (kf 300) norm = 0.0

        # Now test with distinct middle scores:
        hit1_a = MagicMock(id="p1", score=1.0, payload={"video_id": "V1", "keyframe_idx": 100}) # max -> 1.0
        hit2_a = MagicMock(id="p2", score=0.8, payload={"video_id": "V1", "keyframe_idx": 200}) # middle -> (0.8-0.0)/1.0 = 0.8
        hit_min_a = MagicMock(id="p_min", score=0.0, payload={"video_id": "V1", "keyframe_idx": 999}) # min -> 0.0

        hit1_b = MagicMock(id="p1", score=10.0, payload={"video_id": "V1", "keyframe_idx": 100}) # max -> 1.0
        hit3_b = MagicMock(id="p3", score=6.0, payload={"video_id": "V1", "keyframe_idx": 300}) # middle -> (6.0-0.0)/10.0 = 0.6
        hit_min_b = MagicMock(id="p_min", score=0.0, payload={"video_id": "V1", "keyframe_idx": 999}) # min -> 0.0

        def mock_query_points2(collection_name, query, using, limit, search_params, with_payload):
            mock_res = MagicMock()
            if using == "SigLIP2":
                mock_res.points = [hit1_a, hit2_a, hit_min_a]
            elif using == "Qwen3_VL_Embedding":
                mock_res.points = [hit1_b, hit3_b, hit_min_b]
            return mock_res

        system.qdrant_client.query_points.side_effect = mock_query_points2

        with patch.dict(EMBEDDING_WEIGHTS, {"SigLIP2": 1.0, "Qwen3_VL_Embedding": 1.0}):
            results2 = system.semantic_search(
                "test query 2",
                model_names=["SigLIP2", "Qwen3_VL_Embedding"],
                top_k=10,
                limit=10
            )

        res_by_kf2 = {r["keyframe_index"]: r["score"] for r in results2}

        # Keyframe 200 (p2) is ONLY in SigLIP2: norm_score = 0.8
        # Because only SigLIP2 returned it, score MUST be 0.8 / 1.0 = 0.8 (NOT 0.8 / 2 = 0.4!)
        self.assertAlmostEqual(res_by_kf2[200], 0.8, places=4)

        # Keyframe 300 (p3) is ONLY in Qwen3_VL: norm_score = 0.6
        # Because only Qwen3_VL returned it, score MUST be 0.6 / 1.0 = 0.6 (NOT 0.6 / 2 = 0.3!)
        self.assertAlmostEqual(res_by_kf2[300], 0.6, places=4)

        # Keyframe 100 (p1) is in BOTH models:
        # SigLIP2 norm = 1.0, Qwen3_VL norm = 1.0
        # Score = (1.0*1.0 + 1.0*1.0) / (1.0 + 1.0) = 1.0
        self.assertAlmostEqual(res_by_kf2[100], 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
