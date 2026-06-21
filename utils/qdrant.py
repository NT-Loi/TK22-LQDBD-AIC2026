from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import logging

logger = logging.getLogger(__name__)

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