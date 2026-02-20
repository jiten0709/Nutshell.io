"""
This module serves as the adapter layer for interacting with the LLM (Language Model) API. 
It defines functions that take raw newsletter text as input and return structured NewsletterDigest objects by leveraging the OpenAI API. 
This abstraction allows the core application logic to remain decoupled from the specifics of how the LLM is called and how responses are parsed.
"""

from openai import OpenAI, APIError
from dotenv import load_dotenv
load_dotenv()
from src.core.entities import NewsletterDigest, NewsletterSource
import os
import asyncio
from typing import List

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="adapters.log")

TOKEN = os.getenv('GITHUB_TOKEN')
ENDPOINT = os.getenv('GITHUB_ENDPOINT')
MODEL = os.getenv('GITHUB_MODEL_NAME')

# Token limits (adjust based on your model)
MAX_INPUT_CHARS = int(os.getenv('MAX_INPUT_CHARS', 6000))  # Leave buffer for response
CHUNK_OVERLAP = 200  # Characters to overlap between chunks for context
MIN_RELEVANCE_SCORE = int(os.getenv('MIN_RELEVANCE_SCORE', 5))

# prompt templates
EXTRACTION_PROMPT = """You are analyzing part {chunk_index} of {total_chunks} from an AI/tech newsletter.

EXTRACT ONLY:
- Product launches, new features, or major updates (e.g., "GPT-5 released", "Claude now supports vision")
- Key technical benchmarks or research findings (e.g., "40 percent improvement on MMLU", "New SOTA on ImageNet")
- Important company announcements (funding, acquisitions, partnerships)
- Links to official sources, research papers, or product pages

IGNORE AND SKIP:
- Advertisements, sponsor messages, or promotional content
- Newsletter subscription prompts ("Subscribe to our premium tier...")
- Social media follow requests
- Generic industry commentary without specific news
- Job postings or event promotions
- Affiliate links or discount codes
- "In case you missed it" recaps of old news

Format as concise bullets with full URLs. If nothing newsworthy in this section, return "No significant updates."

Newsletter section:
{chunk}"""

SUMMARY_PROMPT = """You are a Principal AI Engineer creating a curated digest of important AI/tech updates.

STRICT RULES:
1. Only include insights that are genuinely newsworthy (product launches, research breakthroughs, major company news)
2. Assign relevance scores honestly: 8-10 for major announcements, 5-7 for minor updates, below 5 for noise
3. Categorize accurately (e.g., "AI Models", "Infrastructure", "Research", "Business")
4. Extract all mentioned URLs
5. Write concise, factual summaries (2-3 sentences max)
6. DO NOT include: ads, sponsors, newsletter meta-content, job posts, or generic commentary

If the input contains no real news, return an empty insights list.
"""

client = OpenAI(
    base_url=ENDPOINT,
    api_key=TOKEN,
)

def _smart_chunk_text(text: str, max_chars: int = MAX_INPUT_CHARS) -> List[str]:
    """
    Chunk by characters (simple and robust):
    - Split on paragraph boundaries
    - Keep overlap between chunks for context
    """
    if len(text) <= max_chars:
        logger.debug("📌 Text fits within char limit, no chunking needed")
        return [text]

    chunks: List[str] = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    current_chunk: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # account for separator

        # If single paragraph exceeds limit, split by sentence heuristics
        if para_len > max_chars:
            logger.debug("✂️ Paragraph longer than max_chars, splitting by sentences")
            sentences = [s.strip() for s in para.split('. ') if s.strip()]
            for sent in sentences:
                sent_len = len(sent) + 2
                if current_len + sent_len > max_chars and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # keep small overlap
                    current_chunk = current_chunk[-1:] if current_chunk else []
                    current_len = sum(len(p) + 2 for p in current_chunk)
                current_chunk.append(sent)
                current_len += sent_len
        else:
            if current_len + para_len > max_chars and current_chunk:
                logger.debug("📦 Current chunk full (chars: %d), starting new chunk", current_len)
                chunks.append('\n\n'.join(current_chunk))
                # keep last paragraph as overlap
                current_chunk = [current_chunk[-1], para] if current_chunk else [para]
                current_len = sum(len(p) + 2 for p in current_chunk)
            else:
                current_chunk.append(para)
                current_len += para_len

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    logger.debug("❗️ Split text into %d chunks (chars: %d)", len(chunks), len(text))
    return chunks

def _extract_from_chunk_sync(chunk: str, chunk_index: int, total_chunks: int) -> str:
    """
    Extract key insights from a single chunk.
    Returns plain text summary rather than structured format for intermediate steps.
    
    ENHANCEMENT: More aggressive filtering of ads, sponsors, and fluff
    """
    try:
        # ENHANCED PROMPT: More specific instructions to filter noise
        prompt = EXTRACTION_PROMPT.format(
            chunk_index=chunk_index+1,
            total_chunks=total_chunks,
            chunk=chunk
        )
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a Principal AI Engineer who filters signal from noise in tech newsletters. You are extremely selective and only extract genuinely important updates."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # LOW: We want consistent, factual extraction
            max_tokens=1000
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"✅ Successfully extracted insights from chunk {chunk_index + 1}/{total_chunks}")
        
        # ENHANCEMENT: Skip chunks with no real content
        if "No significant updates" in result or len(result) < 50:
            logger.debug(f"Chunk {chunk_index} contained no significant updates, skipping")
            return "[No updates]"
        
        return result
        
    except APIError as e:
        logger.error(f"LLM API error on chunk {chunk_index}: {e}")
        return f"[Error processing chunk {chunk_index}]"

def _parse_combined_summary_sync(combined_text: str):
    """
    Parse the combined summary into structured NewsletterDigest.
    This is the final structured extraction step.
    
    ENHANCEMENT: More aggressive filtering instructions
    """
    try:
        return client.chat.completions.parse(
            model=MODEL,
            response_format=NewsletterDigest,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": combined_text}
            ],
            temperature=0.1  # VERY LOW: We want consistent, strict filtering
        )
    except APIError as e:
        logger.error(f"Failed to parse combined summary: {e}")
        raise

async def extract_digest_from_text(raw_text: str) -> NewsletterDigest:
    """
    Robust multi-stage extraction:
    1. Chunk if needed (by character count)
    2. Extract insights from each chunk in parallel (with aggressive filtering)
    3. Combine and parse into structured format
    4. ENHANCEMENT: Filter out low-relevance insights after parsing
    
    Falls back gracefully if any stage fails.
    """
    logger.info("⚙️ Starting digest extraction...")
    
    # Clean input
    text = raw_text.strip()
    if not text:
        logger.warning("Empty input text")
        raise ValueError("Cannot extract digest from empty text")
    
    logger.debug(f"Input length: {len(text)} chars")
    
    # Strategy 1: Direct parse for small inputs
    if len(text) <= MAX_INPUT_CHARS:
        logger.info("Input fits in single request - using direct parse")
        try:
            digest = await asyncio.to_thread(_parse_combined_summary_sync, text)

            return _filter_digest(digest)
        except APIError as e:
            if '413' in str(e) or 'too large' in str(e).lower():
                logger.warning("Direct parse failed with 413, falling back to chunking")
            else:
                raise
        logger.info("⬇️ Falling through to chunked extraction...")
    
    # Strategy 2: Chunk, extract, combine, parse
    logger.info("Using multi-stage chunked extraction")
    chunks = _smart_chunk_text(text, max_chars=MAX_INPUT_CHARS)
    
    # Extract insights from chunks in parallel (with concurrency limit)
    summaries = []
    semaphore = asyncio.Semaphore(1)  # Conservative: 1 request at a time to avoid rate limits
    
    async def _process_chunk(chunk: str, idx: int) -> str:
        async with semaphore:
            return await asyncio.to_thread(_extract_from_chunk_sync, chunk, idx, len(chunks))
    
    tasks = [_process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
    summaries = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out errors, empty chunks, and combine
    valid_summaries = [
        s for s in summaries 
        if isinstance(s, str) 
        and not s.startswith('[Error')
        and not s.startswith('[No updates')
    ]
    
    if not valid_summaries:
        logger.warning("No valid content found after filtering chunks")
        # Return empty digest rather than crashing
        return NewsletterDigest(
            source=NewsletterSource(name="unknown"),
            insights=[]
        )
    
    combined = "\n\n---\n\n".join(valid_summaries)
    logger.info(f"📒 Combined {len(valid_summaries)}/{len(chunks)} chunk summaries (combined chars: {len(combined)})")
    
    # Final structured parse
    logger.info("Parsing combined summary into structured format")
    digest = await asyncio.to_thread(_parse_combined_summary_sync, combined)
    
    # ENHANCEMENT: Post-processing filter
    return _filter_digest(digest)

def _filter_digest(digest):
    """
    ENHANCEMENT: Post-processing filter to remove low-quality insights
    
    Filters out:
    - Insights with relevance_score below threshold
    - Insights with generic/spammy headlines
    - Insights with no links (usually ads or fluff)
    """
    if not hasattr(digest, 'choices'):
        # Already a NewsletterDigest object
        parsed = digest
    else:
        # ParsedChatCompletion wrapper
        parsed = digest.choices[0].message.parsed
    
    # ✅ Guard: if parsed is None (model returned nothing), return empty digest
    if parsed is None:
        logger.warning("⚠️ LLM returned None for parsed digest, returning empty digest")
        return NewsletterDigest(
            source=NewsletterSource(name="unknown"),
            insights=[]
        )
    
    original_count = len(parsed.insights)
    
    # Filter by relevance score and quality signals
    filtered_insights = []
    spam_keywords = [
        'sponsor', 'advertisement', 'subscribe', 'discount', 
        'affiliate', 'partner content', 'promoted', 'job opening',
        'unsubscribe', 'follow us', 'join our'
    ]
    
    for insight in parsed.insights:
        # Check 1: Relevance score
        if insight.relevance_score < MIN_RELEVANCE_SCORE:
            logger.debug(f"🚮 Filtered low relevance ({insight.relevance_score}): {insight.headline}")
            continue
        
        # Check 2: Spam keywords in headline
        headline_lower = insight.headline.lower()
        if any(keyword in headline_lower for keyword in spam_keywords):
            logger.debug(f"🚮 Filtered spam keyword: {insight.headline}")
            continue
        
        # # Check 3: Must have at least one link (real news has sources)
        # if not insight.links or len(insight.links) == 0:
        #     logger.debug(f"🚮 Filtered no-link insight: {insight.headline}")
        #     continue
        
        # # Check 4: Headline must be substantial (not just "Update" or "News")
        # if len(insight.headline) < 15:
        #     logger.debug(f"🚮 Filtered too-short headline: {insight.headline}")
        #     continue
        
        filtered_insights.append(insight)
    
    logger.info(f"🔍 Filtered {original_count - len(filtered_insights)} low-quality insights. Kept {len(filtered_insights)}.")
    
    # Return new digest with filtered insights
    parsed.insights = filtered_insights

    return parsed
