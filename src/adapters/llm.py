from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from src.core.entities import NewsletterDigest
import os

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="adapters.log")

TOKEN = os.getenv('GITHUB_TOKEN')
ENDPOINT = os.getenv('GITHUB_ENDPOINT')
MODEL = os.getenv('GITHUB_MODEL_NAME')

client = OpenAI(
        base_url=ENDPOINT,
        api_key=TOKEN,
    )

async def extract_digest_from_text(raw_text: str) -> NewsletterDigest:
    logger.info("⚙️ Extracting digest from raw text...")
    logger.debug(f"Extracting digest from text of length {len(raw_text)}")
    return client.chat.completions.parse(
        model=MODEL,
        response_format=NewsletterDigest,
        messages=[
            {"role": "system", "content": "You are a Principal AI Engineer. Extract the most high-signal updates from this newsletter content."},
            {"role": "user", "content": raw_text},
        ],
    )