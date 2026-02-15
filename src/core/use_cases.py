from src.adapters.llm import extract_digest_from_text
from src.adapters.vector_store import VectorService

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="core.log")

vs = VectorService()

async def process_new_email(payload: dict):
    email_body = payload.get("TextBody", "") # Mocking the email body for now
    digest = await extract_digest_from_text(email_body)
    
    for incoming in digest.insights:
        existing_id = vs.find_duplicate(incoming.headline)
        
        if existing_id:
            logger.info(f"🔍 Found duplicate for headline: {incoming.headline}. Merging insights...")

            # 1. Fetch current state
            current_payload = vs.get_payload(existing_id)
            
            # 2. Merge Links (Avoid duplicates)
            updated_links = list(set(current_payload.get("links", []) + incoming.links))
            
            # 3. Update Metadata (Add the new source to the history)
            sources = current_payload.get("sources", ["Original Source"])
            new_source = payload.get("From", "Unknown Newsletter")
            if new_source not in sources:
                sources.append(new_source)

            # 4. Patch the record
            vs.patch_payload(existing_id, {
                "links": updated_links,
                "sources": sources,
                "mention_count": current_payload.get("mention_count", 1) + 1
            })
            logger.info(f"🔥 Merged insight: {incoming.headline} (Total sources: {len(sources)})")
        else:
            # New insight logic
            logger.info(f"✨ No duplicate found for headline: {incoming.headline}. Adding as new insight.")
            data = incoming.model_dump()
            data["sources"] = [payload.get("From", "Initial Source")]
            data["mention_count"] = 1
            vs.upsert_insight(data, incoming.headline)
            logger.info(f"✅ Added new insight: {incoming.headline} from source: {payload.get('From', 'Unknown')}")