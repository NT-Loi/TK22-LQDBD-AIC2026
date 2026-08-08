from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, TextIndexParams, TokenizerType
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def setup_text_indexes(client: QdrantClient, collection_name: str, text_fields: list = None):
    """
    Creates full-text indexes on specified payload fields for keyword search.
    Uses MULTILINGUAL tokenizer for Vietnamese text support.
    """
    if text_fields is None:
        text_fields = ["whisper_text"]  # Add "ocr_text" later

    for field_name in text_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.MULTILINGUAL,
                    min_token_len=2,
                    lowercase=True,
                )
            )
            logger.info(f"Created text index on '{field_name}' in collection '{collection_name}'.")
        except Exception as e:
            logger.warning(f"Could not create text index on '{field_name}': {e}")