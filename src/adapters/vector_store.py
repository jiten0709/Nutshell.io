"""
This module defines the VectorService class, which serves as an adapter to interact with the Qdrant vector database and Fastembed for text embeddings. 
It provides methods for finding duplicates, upserting insights, retrieving payloads, and patching payloads. 
This abstraction allows the core application logic to remain decoupled from the specifics of the vector database and embedding service.

- Qdrant: It's a powerful vector database featuring efficient similarity search and storage for high-dimensional data, payload filtering, snapshots and backup, 1:1 production parity.
- Fastembed: It keeps your costs at 0 for the PoC while maintaining high accuracy for short text snippets like headlines.
"""

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from typing import Optional
import uuid
from dotenv import load_dotenv
load_dotenv()
import os

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="adapters.log")

THRESHOLD = float(os.getenv("THRESHOLD", 0.85))

class VectorService:
    def __init__(self, collection_name: str = "nutshell"):
        self.client = QdrantClient(host="localhost", port=6333)
        self.encoder = TextEmbedding() # Defaults to BAAI/bge-small-en-v1.5
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Initialize the collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384, # Size for bge-small-en
                    distance=models.Distance.COSINE
                )
            )

    def find_duplicate(self, text: str, threshold: float = THRESHOLD) -> Optional[str]:
        """Returns the ID of a similar news item if it exists above the threshold."""
        vector = list(self.encoder.embed([text]))[0]
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=1,
        ).points

        logger.debug(f"find_duplicate: Queried for text '{text[:20]}...' and got results: {results}")
        
        if results and results[0].score >= threshold:
            return results[0].id
        return None

    def upsert_insight(self, insight_data: dict, text_for_vector: str):
        """
        The 'Write' path: Creates a brand new point with a vector.
        """
        vector = list(self.encoder.embed([text_for_vector]))[0]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=insight_data
                )
            ]
        )

        logger.debug(f"upsert_insight: Upserted insight with headline '{text_for_vector}' and data: {insight_data}")

    def get_payload(self, point_id: str) -> dict:
        """
        Retrieves the metadata for an existing point.
        Used to see what sources/links we already have.
        """
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id]
        )

        logger.debug(f"get_payload: Retrieved payload for point_id '{point_id}': {result}")

        return result[0].payload if result else {}
    
    def patch_payload(self, point_id: str, new_data: dict):
        """
        The 'Update' path: Only modifies specific keys in the metadata.
        Crucial for the 'Merging' logic.
        """
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=new_data,
            points=[point_id]
        )
        logger.debug(f"patch_payload: Patched payload for point_id '{point_id}' with new_data: {new_data}")    