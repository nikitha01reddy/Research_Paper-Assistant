import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

class AnalysisAgent:
    """
    Analysis Agent
    --------------
    Analyzes paper abstracts to extract Title, Authors, Abstract, Keywords, and Main Contribution.
    """
    def __init__(self):
        # Configure Gemini Client using the new google-genai SDK
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file")
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"

    def analyze_paper(self, paper: dict) -> dict:
        """
        Analyzes the paper title and abstract and returns structured info.
        """
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        prompt = f"""
You are the Paper Analysis Agent for an AI Research Paper Assistant.
Read the following research paper title and abstract, then extract key metadata.

Title: {title}
Abstract: {abstract}

Respond ONLY with a JSON object in this format (no markdown code fence blocks, just raw JSON):
{{
  "title": "Title of the paper",
  "authors": "Authors list if mentioned or 'Not specified'.",
  "abstract": "The abstract text",
  "keywords": ["keyword 1", "keyword 2", "keyword 3"],
  "main_contribution": "1-2 sentence description of the primary contribution of this paper."
}}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            content = response.text.strip()
            # Parse out any markdown wrappers if present
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
            print(f"AnalysisAgent Exception: {e}")
            # Fallback dictionary if parsing or API fails
            return {
                "title": title,
                "authors": "Not specified",
                "abstract": abstract,
                "keywords": ["Machine Learning"],
                "main_contribution": "Analysis failed, but the paper addresses topic modeling/evaluation in Machine Learning."
            }
