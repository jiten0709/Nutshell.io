"""
This script connects to the Nylas API, fetches recent newsletters, and processes only the new ones. 
It uses an EmailTracker to keep track of which emails have already been processed, ensuring that it doesn't waste resources on duplicates. 
The script also logs its actions for easy monitoring and debugging. 
To run this script, use the command: `python scripts/sync_inbox.py` or `PYTHONPATH=. python scripts/sync_inbox.py` from the root of the project.
"""

import asyncio
from src.adapters.mail import NylasAdapter
from src.adapters.email_tracker import EmailTracker
from src.core.use_cases import process_new_email
from src.adapters.vector_store import VectorService

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="scripts.log")

async def sync():
    # Check Qdrant connection first
    logger.info("🔍 Checking Qdrant connection...")
    try:
        vs = VectorService()
        logger.info(f"🗃️ Qdrant connected. Collection: {vs.collection_name}")
    except Exception as e:
        logger.error(f"❌ Cannot connect to Qdrant: {e}")
        logger.error("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        return
    
    logger.info("📬 Connecting to Nylas...")
    mail_adapter = NylasAdapter()
    tracker = EmailTracker()
    
    logger.info(f"📊 Already processed {tracker.get_count()} emails")
    logger.info("🔍 Searching for new AI newsletters...")
    newsletters = await mail_adapter.get_latest_newsletters()
    
    if not newsletters:
        logger.info("📭 No new newsletters found.")
        return

    new_count = 0
    skipped_count = 0

    for nl in newsletters:
        email_id = nl['id']
        
        # Skip if already processed
        if tracker.is_processed(email_id):
            logger.debug(f"⏭️  Skipping already processed: {nl['subject']}")
            skipped_count += 1
            continue
        
        logger.info(f"📥 Processing NEW: {nl['subject']} from {nl['from']}")
        
        # Map Nylas keys to our process_new_email expectation
        payload = {
            "TextBody": nl['body'],
            "From": nl['from'],
            "Subject": nl['subject'],
            "MessageID": email_id,
            "date": str(nl['date'])
        }
        
        try:
            await process_new_email(payload)
            tracker.mark_processed(email_id)
            new_count += 1
            logger.info(f"✅ Successfully processed: {nl['subject']}")
        except Exception as e:
            logger.error(f"❌ Failed to process {nl['subject']}: {e}")
    
    logger.info(f"✅ Sync complete! Processed {new_count} new emails, skipped {skipped_count} already processed.")
    logger.info("🔄 Refresh your Streamlit dashboard!")

if __name__ == "__main__":
    asyncio.run(sync())
