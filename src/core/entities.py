"""
This file defines the core entities for Nutshell.io, which include NewsletterSource, IntelligenceNode, and NewsletterDigest. 
These entities are used throughout the application to represent the structured data extracted from raw newsletter content. 

- NewsletterSource captures the origin of the news
- IntelligenceNode represents a single piece of structured news with a headline, summary, relevance score, category, and links
- NewsletterDigest aggregates multiple IntelligenceNodes along with metadata about the source and processing time. 

By defining these entities in a centralized module, we ensure consistency and clarity across the application when handling the data extracted from newsletters.
"""

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