import unittest
from elasticsearch import Elasticsearch
from config import ES_HOST_URL, ES_CAPTION_INDEX_NAME
from utils.es import search_caption_bm25

class TestCaptionSearch(unittest.TestCase):
    def test_es_caption_index_exists(self):
        es = Elasticsearch(ES_HOST_URL, request_timeout=30)
        self.assertTrue(es.indices.exists(index=ES_CAPTION_INDEX_NAME))

    def test_search_caption_bm25(self):
        es = Elasticsearch(ES_HOST_URL, request_timeout=30)
        hits = search_caption_bm25(es, ES_CAPTION_INDEX_NAME, "tập thể dục ngoài trời", size=10)
        self.assertGreater(len(hits), 0)
        first = hits[0]
        self.assertIn("video_id", first)
        self.assertIn("shot_idx", first)
        self.assertIn("bm25_score", first)
        self.assertGreater(first["bm25_score"], 0)

if __name__ == "__main__":
    unittest.main()
