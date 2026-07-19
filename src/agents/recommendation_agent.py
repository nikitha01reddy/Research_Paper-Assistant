import os
import sys

# Add parent directory to path so we can import search_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_engine import PaperSearchEngine

class RecommendationAgent:
    """
    Recommendation Agent
    --------------------
    Recommends similar research papers from the FAISS database based on a selected paper.
    """
    def __init__(self, engine: PaperSearchEngine = None):
        self.engine = engine if engine is not None else PaperSearchEngine(load_summarizer=False, load_keybert=False)

    def recommend_similar_papers(self, target_title: str, target_abstract: str, num_recommendations: int = 3) -> list[dict]:
        """
        Uses the target paper's details to query the FAISS database for similar papers,
        excluding the target paper itself.
        """
        # Retrieve slightly more papers than needed to account for filtering the original paper
        search_query = f"{target_title} {target_abstract}"
        results = self.engine.search(search_query, k=num_recommendations + 3)

        recommendations = []
        for paper in results:
            # Check if this paper is the target paper itself to avoid self-recommendation
            if paper["title"].lower().strip() == target_title.lower().strip():
                continue
            
            recommendations.append({
                "title": paper["title"],
                "score": paper["score"],
                "abstract": paper["abstract"]
            })
            
            if len(recommendations) >= num_recommendations:
                break

        return recommendations
