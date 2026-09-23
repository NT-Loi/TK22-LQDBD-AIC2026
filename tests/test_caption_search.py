import pytest
from elasticsearch import Elasticsearch
from config import ES_HOST_URL, ES_CAPTION_INDEX_NAME
from utils.es import search_caption_bm25

def test_es_caption_index_exists():
    es = Elasticsearch(ES_HOST_URL, request_timeout=30)
    assert es.indices.exists(index=ES_CAPTION_INDEX_NAME)

def test_search_caption_bm25():
    es = Elasticsearch(ES_HOST_URL, request_timeout=30)
    hits = search_caption_bm25(es, ES_CAPTION_INDEX_NAME, "tập thể dục ngoài trời", size=10)
    assert len(hits) > 0
    first = hits[0]
    assert "video_id" in first
    assert "shot_idx" in first
    assert "bm25_score" in first
    assert first["bm25_score"] > 0
