import unittest
from unittest.mock import MagicMock, patch
from utils.translator import is_english_query, translate_query_to_vi_if_needed
from retrieval_system import RetrievalSystem

class TestTranslation(unittest.TestCase):
    def test_language_detection(self):
        # English queries
        self.assertTrue(is_english_query("a group of people exercising outdoors, touching their toes"))
        self.assertTrue(is_english_query("two cars colliding on a highway"))
        self.assertTrue(is_english_query("drone view of a dam"))
        self.assertTrue(is_english_query("close-up of a chef cutting vegetables"))
        self.assertTrue(is_english_query("red car driving"))

        # Vietnamese queries with accents
        self.assertFalse(is_english_query("người mặc áo đỏ đang phát biểu"))
        self.assertFalse(is_english_query("Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục"))
        self.assertFalse(is_english_query("con đập được quay từ trên cao"))

        # Vietnamese queries without accents
        self.assertFalse(is_english_query("nguoi mac ao do dang phat bieu"))
        self.assertFalse(is_english_query("nhom nguoi tap the duc ngoai troi"))

    def test_translate_query_vi_noop(self):
        vi_query = "cảnh quay bờ biển lúc hoàng hôn"
        res, was_trans = translate_query_to_vi_if_needed(vi_query)
        self.assertFalse(was_trans)
        self.assertEqual(res, vi_query)

    @patch("utils.translator._get_gemini_client")
    def test_caption_search_auto_translates_english(self, mock_client_getter):
        # Clear lru_cache to ensure test isolation
        from utils.translator import translate_en_to_vi
        translate_en_to_vi.cache_clear()

        # Mock Gemini client
        mock_response = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "một nhóm người đang tập thể dục ngoài trời"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_getter.return_value = mock_client

        import numpy as np
        system = RetrievalSystem.__new__(RetrievalSystem)
        system.es_client = MagicMock()
        system.qdrant_client = MagicMock()
        mock_encoder = MagicMock()
        mock_encoder.return_value = np.zeros((1, 1024))
        system.text_encoders = {"Qwen3_Embedding": mock_encoder}

        # Mock Qdrant and ES responses
        system.qdrant_client.query_points.return_value.points = []
        with patch("config.AUTO_TRANSLATE_EN_CAPTION", True), \
             patch("retrieval_system.search_caption_bm25", return_value=[]):
            results = system.caption_search("a group of people exercising outdoors", top_k=10)

        # Verify Qwen3 encoder was called with the TRANSLATED text, not English
        mock_encoder.assert_called_once()
        called_arg = mock_encoder.call_args[0][0]
        self.assertIn("tập thể dục", called_arg)

if __name__ == "__main__":
    unittest.main()
