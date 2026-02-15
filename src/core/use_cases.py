from src.adapters.llm import extract_digest_from_text
from src.adapters.vector_store import VectorService

vs = VectorService()

async def process_new_email(payload: dict):
    # 1. Extract body (Adjust key based on your provider, e.g., 'TextBody' for Postmark)
    email_body = payload.get("TextBody", "")
    
    # 2. Extract structured insights via LLM
    digest = await extract_digest_from_text(email_body)
    
    # 3. Handle Deduplication and Storage
    for insight in digest.insights:
        existing_id = vs.find_duplicate(insight.headline)
        
        if existing_id:
            # MOAT FEATURE: Instead of doing nothing, we update the existing cluster 
            # to add the new source URL and maybe update the relevance score.
            print(f"Updating existing cluster: {existing_id}")
        else:
            print(f"Creating new nutshell: {insight.headline}")
            vs.upsert_insight(insight.model_dump(), insight.headline)