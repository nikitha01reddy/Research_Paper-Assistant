import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

class PlannerAgent:
    """
    Planner Agent
    -------------
    Analyzes the user's research query, explains the search strategy,
    and formulates specific search terms for the Search Agent.
    """
    def __init__(self):
        # Configure Gemini Client using the new google-genai SDK
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file")
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"

    def plan(self, query: str) -> dict:
        """
        Generates a search plan and search queries based on the user's input.
        Returns a dictionary with 'explanation' and 'search_queries'.
        """
        prompt = f"""
You are the Planner Agent for an AI Research Paper Assistant.
Your task is to analyze the user's research query and formulate a plan to search the paper database.

User Query: "{query}"

Respond ONLY with a JSON object in the following format (do not include markdown block formatting, just the raw JSON):
{{
  "explanation": "Brief explanation of what the user is looking for and the search strategy.",
  "search_queries": ["query 1", "query 2"]
}}

Make sure to provide 1 to 3 search queries that represent different facets of the user's question, optimized for a semantic vector search. Keep queries concise.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            content = response.text.strip()
            # Remove any markdown code fence wrappers if the model generated them
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            
            data = json.loads(content)
            return data
        except Exception as e:
            # Robust fallback if JSON parsing fails or API fails
            print(f"PlannerAgent Error: {e}")
            return {
                "explanation": f"Searching database for topics related to: {query}.",
                "search_queries": [query]
            }

if __name__ == "__main__":
    # Test query
    planner = PlannerAgent()
    print(planner.plan("Explain residual neural networks and how they help with exploding gradients"))
