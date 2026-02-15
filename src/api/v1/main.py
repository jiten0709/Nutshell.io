"""
This is the main FastAPI application file that defines the API endpoints for Nutshell.io.
"""

from fastapi import FastAPI, Request, BackgroundTasks
from src.core.use_cases import process_new_email

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="api.log")

app = FastAPI(title="Nutshell.io API")

# welcome endpoint
@ app.get("/")
async def root():
    return {"message": "Welcome to Nutshell.io API. Use /webhooks/inbound-email to POST new email content."}

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

    return {"status": "received"}
