"""
This is the main FastAPI application file that defines the API endpoints for Nutshell.io.
"""

from fastapi import FastAPI, Request, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from src.core.use_cases import process_new_email
from src.adapters.vector_store import VectorService
from typing import Optional

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="api_v1.log")

app = FastAPI(title="Nutshell.io API", version="1.0.0")

# welcome endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Nutshell.io API",
        "endpoints": {
            "webhook": "/webhooks/inbound-email",
            "insights": "/api/v1/insights",
            "health": "/health"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "nutshell-api"}

# This is the main webhook endpoint that Postmark/Nylas will call when a new email arrives.
@app.post("/webhooks/inbound-email")
async def handle_inbound_email(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint for Postmark/Nylas to POST new email content.
    We use BackgroundTasks to return a 200 OK immediately and process the AI logic in the background.
    """
    logger.info("📬 Received new inbound email webhook. Offloading to background task...")
    raw_payload = await request.json()
    
    # We offload the heavy LLM/Vector work to the background
    background_tasks.add_task(process_new_email, raw_payload)

    return {"status": "received", "message": "Email processing started in background"}

# endpoint to query insights
@app.get("/api/v1/insights")
async def get_insights(
    limit: int = Query(50, ge=1, le=100),
    category: Optional[str] = None,
    min_relevance: Optional[int] = Query(None, ge=1, le=10),
    tag: Optional[str] = None,
    company: Optional[str] = None
):
    """
    Retrieve insights with optional filtering.
    
    - **limit**: Maximum number of insights to return (1-100)
    - **category**: Filter by category (e.g., 'Model Release', 'Open Source')
    - **min_relevance**: Minimum relevance score (1-10)
    - **tag**: Filter by tag (e.g., 'LLM', 'OpenAI')
    - **company**: Filter by company mentioned
    """
    try:
        vector_service = VectorService()
        
        # Build filter conditions
        filter_conditions = []
        if category:
            filter_conditions.append({"key": "category", "match": {"value": category}})
        if min_relevance:
            filter_conditions.append({"key": "relevance_score", "range": {"gte": min_relevance}})
        if tag:
            filter_conditions.append({"key": "tags", "match": {"any": [tag]}})
        if company:
            filter_conditions.append({"key": "companies_mentioned", "match": {"any": [company]}})
        
        # Query with filters
        from qdrant_client import models
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
        
        points, _ = vector_service.client.scroll(
            collection_name=vector_service.collection_name,
            scroll_filter=query_filter,
            with_payload=True,
            limit=limit
        )
        
        insights = [p.payload for p in points]
        
        return {
            "count": len(insights),
            "insights": insights
        }
    except Exception as e:
        logger.error(f"Error fetching insights: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to fetch insights"})