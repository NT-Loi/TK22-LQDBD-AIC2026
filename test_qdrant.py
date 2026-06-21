from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
res = client.query_points(collection_name="video_frames", query=[0.1]*1024, using="CLIP_H14", limit=1)
print(type(res))
print(dir(res))
try:
    print(res.points)
except Exception as e:
    print("Error:", e)
