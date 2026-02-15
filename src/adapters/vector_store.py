"""
# Qdrant: It's a powerful vector database featuring efficient similarity search and storage for high-dimensional data, payload filtering, snapshots and backup, 1:1 production parity.
# Fastembed: It keeps your costs at 0 for the PoC while maintaining high accuracy for short text snippets like headlines.
"""

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from typing import List, Optional
import uuid

class VectorService:
    def __init__(self, collection_name: str = "nutshells"):
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

    def find_duplicate(self, text: str, threshold: float = 0.85) -> Optional[str]:
        """Returns the ID of a similar news item if it exists above the threshold."""
        vector = list(self.encoder.embed([text]))[0]
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=1,
        ).points
        
        if results and results[0].score >= threshold:
            return results[0].id
        return None

    def upsert_insight(self, insight_data: dict, text_for_vector: str):
        """Inserts a new insight with its metadata."""
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