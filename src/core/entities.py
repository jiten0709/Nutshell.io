from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from datetime import datetime

class NewsletterSource(BaseModel):
    name: str
    url: Optional[str] = None

class IntelligenceNode(BaseModel):
    """A structured piece of AI news extracted from a raw source."""
    headline: str = Field(..., description="A punchy, technical headline.")
    summary: str = Field(..., description="2-3 bullet points of technical substance.")
    relevance_score: int = Field(..., ge=1, le=10, description="How critical is this for a Senior MLE?")
    category: str = Field(..., description="e.g., 'Model Release', 'Open Source', 'Hardware'")
    links: List[str] = Field(default_factory=list)

class NewsletterDigest(BaseModel):
    source: NewsletterSource
    processed_at: datetime = Field(default_factory=datetime.now)
    insights: List[IntelligenceNode]