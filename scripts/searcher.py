from ddgs import DDGS
from typing import List, Dict

try:
    from logger import setup_aether_logger
except:
    from .logger import setup_aether_logger

logger = setup_aether_logger("Searcher")

class SearchEngine:
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        logger.info(f"Searching for: '{query}' (limit: {max_results})")
        results = []
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=max_results)]
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []