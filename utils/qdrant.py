from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from typing import List, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range


def batch_upsert_points(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 500,
    wait: bool = True,
) -> int:
    """
    Upsert a list of PointStruct objects into a Qdrant collection in batches.

    Args:
        client: An initialised QdrantClient instance.
        collection_name: Target Qdrant collection.
        points: Full list of PointStruct objects to upsert.
        batch_size: Number of points per upsert call.
        wait: If True, block until each batch is committed.

    Returns:
        Total number of points upserted.
    """
    total = 0
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=wait,
        )
        total += len(batch)
    return total

def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_sizes: dict, distance: Distance = Distance.COSINE, overwrite=False):
    """
    Sets up a Qdrant collection with multiple named vectors.
    vector_sizes is a dictionary mapping vector names to their dimensionality.
    Example:
        vector_sizes = {"clip": 512, "blip": 768}
    """
    if client.collection_exists(collection_name=collection_name):
        if overwrite:
            logger.info(f"Collection '{collection_name}' already exists. Deleting collection...")
            client.delete_collection(collection_name=collection_name)
        else:
            logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
            return

    logger.info(f"Creating Qdrant collection '{collection_name}' with vectors: {list(vector_sizes.keys())}")
    
    vectors_config = {
        name: VectorParams(size=size, distance=distance)
        for name, size in vector_sizes.items()
    }
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config
    )
    logger.info(f"Collection '{collection_name}' created successfully.")

def build_qdrant_filter(objects: list) -> Optional[Filter]:
        if not objects:
            return None
            
        must_conditions = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            label = obj.get("label")
            if not label:
                continue
                
            must_conditions.append(
                FieldCondition(
                    key="objects",
                    match=MatchValue(value=label)
                )
            )
            
            min_instances = obj.get("min_instances", 1)
            max_instances = obj.get("max_instances")
            
            range_kwargs = {"gte": min_instances}
            if max_instances is not None and isinstance(max_instances, int):
                range_kwargs["lte"] = max_instances
                
            must_conditions.append(
                FieldCondition(
                    key=f"class_counts.{label}",
                    range=Range(**range_kwargs)
                )
            )
            
        if not must_conditions:
            return None
            
        return Filter(must=must_conditions)