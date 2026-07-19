import os
import sys

# Add parent directory to path so we can import search_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_engine import PaperSearchEngine

class SearchAgent:
    """
    Search Agent
    ------------
    Queries the FAISS index for relevant papers using search terms.
    """
    def __init__(self, engine: PaperSearchEngine = None):
        self.engine = engine if engine is not None else PaperSearchEngine(load_summarizer=False, load_keybert=False)

    def search_papers(self, queries: list[str], k: int = 5) -> list[dict]:
        """
        Queries the database with multiple search terms, merges, and deduplicates.
        """
        merged_results = {}
        for query in queries:
            results = self.engine.search(query, k=k)
            for paper in results:
                row_idx = paper["row_index"]
                # Deduplicate based on database index
                if row_idx not in merged_results:
                    merged_results[row_idx] = paper
                else:
                    # Keep the search result with the higher similarity score
                    if paper["score"] > merged_results[row_idx]["score"]:
                        merged_results[row_idx] = paper

        # Sort by similarity score descending
        sorted_papers = sorted(merged_results.values(), key=lambda x: x["score"], reverse=True)

        # Re-rank final output list
        for i, paper in enumerate(sorted_papers):
            paper["rank"] = i + 1

        return sorted_papers[:k]
