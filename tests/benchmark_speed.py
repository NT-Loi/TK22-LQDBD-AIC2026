import time
import requests
import json

BASE_URL = "http://127.0.0.1:8000/search"

# Sample queries
QUERY_VI = "người mặc áo đỏ phát biểu ngoài trời"
QUERY_MULTI_1 = "người đi vào cửa hàng"
QUERY_MULTI_2 = "người mua hàng tại quầy"

benchmarks = [
    {
        "name": "1. Keyframe Search (SigLIP2 single)",
        "payload": {
            "text_queries": [QUERY_VI],
            "models": ["SigLIP2"],
            "search_mode": "keyframe",
            "limit": 100
        }
    },
    {
        "name": "2. Keyframe Search (Qwen3_VL_Embedding single)",
        "payload": {
            "text_queries": [QUERY_VI],
            "models": ["Qwen3_VL_Embedding"],
            "search_mode": "keyframe",
            "limit": 100
        }
    },
    {
        "name": "3. Keyframe Search (Ensemble: SigLIP2 + Qwen3_VL)",
        "payload": {
            "text_queries": [QUERY_VI],
            "models": ["SigLIP2", "Qwen3_VL_Embedding"],
            "search_mode": "keyframe",
            "limit": 100
        }
    },
    {
        "name": "4. Caption Search (Hybrid Dense 8-aspects + BM25 ES)",
        "payload": {
            "text_queries": [QUERY_VI],
            "search_mode": "caption",
            "limit": 100
        }
    },
    {
        "name": "5. Fused Search (Both Keyframe SigLIP2 + Caption)",
        "payload": {
            "text_queries": [QUERY_VI],
            "models": ["SigLIP2"],
            "search_mode": "both",
            "caption_weight": 0.5,
            "limit": 100
        }
    },
    {
        "name": "6. Fused Search (Both Ensemble + Caption)",
        "payload": {
            "text_queries": [QUERY_VI],
            "models": ["SigLIP2", "Qwen3_VL_Embedding"],
            "search_mode": "both",
            "caption_weight": 0.5,
            "limit": 100
        }
    },
    {
        "name": "7. Temporal Search (2 Events, SigLIP2)",
        "payload": {
            "text_queries": [QUERY_MULTI_1, QUERY_MULTI_2],
            "models": ["SigLIP2"],
            "limit": 100
        }
    },
    {
        "name": "8. Keyframe + Text Query Filter (Frame level)",
        "payload": {
            "text_queries": [QUERY_VI],
            "text_filters": [{"text": "xe ô tô", "level": "frame"}],
            "models": ["SigLIP2"],
            "search_mode": "keyframe",
            "limit": 100
        }
    },
    {
        "name": "9. Keyframe + OCR Filter + Audio Filter",
        "payload": {
            "text_queries": [QUERY_VI],
            "ocr_query": [{"text": "Việt Nam", "level": "video"}],
            "audio": [{"text": "chào buổi sáng", "level": "video"}],
            "models": ["SigLIP2"],
            "search_mode": "keyframe",
            "limit": 100
        }
    }
]

print("=" * 75)
print(f"{'BENCHMARK CONFIGURATION':<45} | {'AVG (ms)':>10} | {'MIN (ms)':>10} | {'RESULTS':>8}")
print("=" * 75)

for bench in benchmarks:
    # 1 Warmup run
    try:
        requests.post(BASE_URL, json=bench["payload"], timeout=60)
    except Exception as e:
        print(f"Error on warmup {bench['name']}: {e}")
        continue

    # 3 timed runs
    times = []
    res_count = 0
    for _ in range(3):
        t0 = time.perf_counter()
        resp = requests.post(BASE_URL, json=bench["payload"], timeout=60)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        if resp.status_code == 200:
            res_count = len(resp.json())

    avg_time = sum(times) / len(times)
    min_time = min(times)
    print(f"{bench['name']:<45} | {avg_time:>10.2f} | {min_time:>10.2f} | {res_count:>8}")

print("=" * 75)
