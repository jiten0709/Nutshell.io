"""
This module contains the core use case logic for processing new email content. 
When a new email is received, the process_new_email function is called with the raw payload. 
The function extracts the email body, uses the LLM adapter to get a structured NewsletterDigest, and then checks for duplicates in the vector store. 
If a duplicate headline is found, it merges the insights and updates the existing record. 
If not, it adds a new insight to the vector store. 
This file serves as the central place for the main business logic of how incoming newsletter content is processed and stored.
"""

from src.adapters.llm import extract_digest_from_text
from src.adapters.vector_store import VectorService
from src.core.entities import NewsletterDigest
from datetime import datetime

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="core.log")

vs = VectorService()

async def process_new_email(payload: dict):
    """
    Core use case: Process an inbound newsletter email.
    1. Extract raw text
    2. Extract structured digest via LLM
    3. Check for duplicates in vector DB
    4. Merge or insert new insights with full metadata
    """
    try:
        logger.info("📬 Processing new email...")
        
        # Extract email body (handle both Postmark and Nylas formats)
        email_body = payload.get("TextBody", payload.get("body", ""))
        
        if not email_body:
            logger.warning("⚠️ Empty email body, skipping")
            return
        
        # Extract email metadata (source, subject, date)
        email_source = payload.get("From", payload.get("from", "unknown@unknown.com"))
        email_subject = payload.get("Subject", payload.get("subject", "No Subject"))
        email_date = payload.get("Date", payload.get("date", datetime.utcnow().isoformat()))
        
        # Create source metadata object
        source_metadata = {
            "email": email_source,
            "subject": email_subject,
            "date": str(email_date)  # Ensure it's a string
        }
        
        logger.info(f"📧 Processing: '{email_subject}' from {email_source}")
        
        logger.info("🤖 Extracting digest from email...")
        digest = await extract_digest_from_text(email_body)
        
        newsletter_digest = digest
        if not newsletter_digest or not newsletter_digest.insights:
            logger.warning(f"⚠️ No valid insights extracted from '{email_subject}'. Skipping email.")
            return
        
        logger.info(f"✅ Extracted {len(newsletter_digest.insights)} insights from '{email_subject}'")
        
        # Process each insight from the digest
        for incoming in newsletter_digest.insights:
            logger.debug(f"Processing insight: {incoming.headline}")
            
            # Check for duplicate
            dup_id = vs.find_duplicate(incoming.headline)
            
            if dup_id:
                logger.info(f"🔍 Found duplicate for headline: {incoming.headline}. Merging insights...")

                # 1. Fetch current state
                current_payload = vs.get_payload(dup_id)
                
                # Merge list fields (tags, companies_mentioned, key_people, links)
                merged_links = list(dict.fromkeys(current_payload.get("links", []) + incoming.links))
                merged_tags = list(dict.fromkeys(current_payload.get("tags", []) + incoming.tags))
                merged_companies = list(dict.fromkeys(current_payload.get("companies_mentioned", []) + incoming.companies_mentioned))
                merged_people = list(dict.fromkeys(current_payload.get("key_people", []) + incoming.key_people))
                
                # Update Sources (Store full source metadata, not just email)
                sources = current_payload.get("sources", [])
                source_subjects = [s.get("subject") for s in sources if isinstance(s, dict)]
                if email_subject not in source_subjects:
                    sources.append(source_metadata)
                    logger.debug(f"Added new source: {email_subject}")
                else:
                    logger.debug(f"Source already recorded: {email_subject}")

                # Update relevance score (take the max)
                updated_relevance = max(
                    current_payload.get('relevance_score', 0),
                    incoming.relevance_score
                )
                
                # Track first and last seen dates
                first_seen = current_payload.get("first_seen", email_date)
                last_seen = email_date

                # Patch the record with merged data
                vs.patch_payload(dup_id, {
                    "links": merged_links,
                    "tags": merged_tags,
                    "companies_mentioned": merged_companies,
                    "key_people": merged_people,
                    "sources": sources,
                    "mention_count": len(sources), 
                    "summary": incoming.summary,  # Update with latest summary
                    "relevance_score": updated_relevance,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "category": incoming.category  # Update category in case it changed
                })
                
                logger.info(f"🔥 Merged insight: {incoming.headline}")
                logger.info(f"   - Total sources: {len(sources)}")
                logger.info(f"   - Latest source: '{email_subject}'")
                logger.info(f"   - Links added: {len(set(incoming.links) - set(current_payload.get('links', [])))}")
                logger.info(f"   - Relevance: {updated_relevance}/10")
            else:
                # New insight logic
                logger.info(f"✨ New insight found: {incoming.headline}")
                data = incoming.model_dump()
                
                # Add comprehensive metadata
                data["sources"] = [source_metadata]  # Store full metadata, not just email string
                data["mention_count"] = 1
                data["first_seen"] = str(email_date)
                data["last_seen"] = str(email_date)
                data["original_subject"] = email_subject  # Track which newsletter first mentioned it
                
                vs.upsert_insight(data, incoming.headline)
                logger.info(f"✅ Added new insight from: '{email_subject}'")
                logger.info(f"   - Category: {incoming.category}")
                logger.info(f"   - Relevance: {incoming.relevance_score}/10")
                logger.info(f"   - Links: {len(incoming.links)}")
        
        logger.info(f"🎉 Email processing complete for '{email_subject}'")
        
    except Exception as e:
        logger.error(f"❌ Error processing email '{email_subject}': {e}", exc_info=True)
        raise  # Re-raise to let caller handle