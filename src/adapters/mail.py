from nylas import Client
import os
from dotenv import load_dotenv
load_dotenv()

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="adapters.log")

# env variables
NYLAS_API_KEY = os.getenv("NYLAS_API_KEY")
NYLAS_GRANT_ID = os.getenv("NYLAS_GRANT_ID")
NYLAS_URI = os.getenv("NYLAS_URI")
LIMIT = os.getenv("LIMIT", 5)
SEARCH_QUERY = os.getenv("SEARCH_QUERY", "from:tldr.ai OR from:theneuron.ai")

class NylasAdapter:
    def __init__(self):
        self.nylas = Client(
            api_key=NYLAS_API_KEY,
            api_uri=NYLAS_URI
        )
        self.grant_id = NYLAS_GRANT_ID
        self.search_query = SEARCH_QUERY

    async def get_latest_newsletters(self, limit: int = LIMIT):
        """Fetches the most recent newsletters based on sender filters."""
        # You can filter by 'from' to only get newsletters
        # e.g., 'dan@tldr.tech', 'pete@theneuron.ai'
        raw = self.nylas.messages.list(
            self.grant_id,
            query_params={
                "limit": limit,
                "search_query_native": self.search_query
            }
        )
        
        if isinstance(raw, (tuple, list)):
            if len(raw) >= 1:
                messages = raw[0]
            else:
                messages = []
        else:
            messages = raw

        # Ensure messages is iterable/list
        messages = list(messages) if not isinstance(messages, list) else messages

        newsletter_data = []
        for msg in messages:
            newsletter_data.append({
                "id": msg.id,
                "from": msg.from_[0]['email'],
                "subject": msg.subject,
                "body": msg.body, # Nylas gives you the full HTML/Text
                "date": msg.date
            })
        logger.info(f"🔎 Fetched {len(newsletter_data)} newsletters from Nylas.")

        return newsletter_data