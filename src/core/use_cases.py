"""
about this file: This module contains the core use case logic for processing new email content. 
When a new email is received, the process_new_email function is called with the raw payload. 
The function extracts the email body, uses the LLM adapter to get a structured NewsletterDigest, and then checks for duplicates in the vector store. 
If a duplicate headline is found, it merges the insights and updates the existing record. 
If not, it adds a new insight to the vector store. 
This file serves as the central place for the main business logic of how incoming newsletter content is processed and stored.
"""

from src.adapters.llm import extract_digest_from_text
from src.adapters.vector_store import VectorService

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="core.log")

vs = VectorService()

async def process_new_email(payload: dict):
    """
    Core use case: Process an inbound newsletter email.
    1. Extract raw text
    2. Extract structured digest via LLM
    3. Check for duplicates in vector DB
    4. Merge or insert new insights
    """
    logger.info("📬 Processing new email...")
    
    # Extract email body (handle both Postmark and Nylas formats)
    email_body = payload.get("TextBody", payload.get("body", ""))
    
    if not email_body:
        logger.warning("Empty email body, skipping")
        return
    
    logger.info("🤖 Extracting digest from email...")
    digest = await extract_digest_from_text(email_body)
    
    # FIX: Access the parsed NewsletterDigest object correctly
    newsletter_digest = digest.choices[0].message.parsed
    
    # Get source information
    email_source = payload.get("From", payload.get("from", "Unknown Newsletter"))
    
    # Process each insight from the digest
    for incoming in newsletter_digest.insights:
        logger.debug(f"Processing insight: {incoming.headline}")
        
        # Check for duplicate
        dup_id = vs.find_duplicate(incoming.headline)
        
        if dup_id:
            logger.info(f"🔍 Found duplicate for headline: {incoming.headline}. Merging insights...")

            # 1. Fetch current state
            current_payload = vs.get_payload(dup_id)
            
            # 2. Merge Links (Avoid duplicates)
            existing_links = set(current_payload.get("links", []))
            new_links = set(incoming.links)
            merged_links = list(existing_links | new_links)
            
            # 3. Update Sources (Add the new source to the history)
            sources = current_payload.get("sources", ["Original Source"])
            if email_source not in sources:
                sources.append(email_source)

            # 4. Update relevance score (take the max)
            updated_relevance = max(
                current_payload.get('relevance_score', 0),
                incoming.relevance_score
            )

            # 5. Patch the record with merged data
            vs.patch_payload(dup_id, {
                "links": merged_links,
                "sources": sources,
                "mention_count": current_payload.get("mention_count", 1) + 1,
                "summary": incoming.summary,  # Update with latest summary
                "relevance_score": updated_relevance
            })
            
            logger.info(f"🔥 Merged insight: {incoming.headline}")
            logger.info(f"   - Total sources: {len(sources)}")
            logger.info(f"   - Mentions: {current_payload.get('mention_count', 1) + 1}")
            logger.info(f"   - Links added: {len(new_links - existing_links)}")
        else:
            # New insight logic
            logger.info(f"✨ New insight found: {incoming.headline}")
            data = incoming.model_dump()
            data["sources"] = [email_source]
            data["mention_count"] = 1
            vs.upsert_insight(data, incoming.headline)
            logger.info(f"✅ Added new insight from source: {email_source}")
    
    logger.info("✅ Email processing complete")