import asyncio
from src.adapters.mail import NylasAdapter
from src.core.use_cases import process_new_email

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="scripts.log")

async def sync():
    logger.info("📬 Connecting to Nylas...")
    mail_adapter = NylasAdapter()
    
    logger.info("🔍 Searching for new AI newsletters...")
    newsletters = await mail_adapter.get_latest_newsletters()
    
    if not newsletters:
        logger.info("📭 No new newsletters found.")
        return

    for nl in newsletters:
        logger.debug(f"📥 Processing: {nl['subject']} from {nl['from']}")
        # Map Nylas keys to our process_new_email expectation
        payload = {
            "TextBody": nl['body'],
            "From": nl['from'],
            "Subject": nl['subject']
        }
        await process_new_email(payload)
    
    logger.info("✅ Sync complete. Refresh your Streamlit dashboard!")

if __name__ == "__main__":
    asyncio.run(sync())