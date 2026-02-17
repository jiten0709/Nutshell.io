"""
Tracks processed emails to avoid reprocessing.
Uses a simple JSON file to store email IDs that have been processed.
"""
import json
from pathlib import Path
from typing import Set

class EmailTracker:
    def __init__(self, tracking_file: str = "data/processed_emails.json"):
        self.tracking_file = Path(tracking_file)
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        self.processed_ids: Set[str] = self._load()

    def _load(self) -> Set[str]:
        """Load processed email IDs from file."""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                data = json.load(f)
                return set(data.get('processed_ids', []))
        return set()

    def _save(self):
        """Save processed email IDs to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump({'processed_ids': list(self.processed_ids)}, f, indent=2)

    def is_processed(self, email_id: str) -> bool:
        """Check if email has been processed."""
        return email_id in self.processed_ids

    def mark_processed(self, email_id: str):
        """Mark email as processed."""
        self.processed_ids.add(email_id)
        self._save()

    def get_count(self) -> int:
        """Get count of processed emails."""
        return len(self.processed_ids)
    