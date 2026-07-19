import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class SummaryAgent:
    """
    Summary Agent
    -------------
    Generates a clear and concise summary of a paper's abstract and findings.
    """
    def __init__(self):
        # Configure Gemini Client using the new google-genai SDK
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file")
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"

    def summarize_paper(self, title: str, abstract: str, main_contribution: str = None) -> str:
        """
        Generates a concise 3-4 sentence summary of the paper.
        """
        contribution_context = f"\nMain Contribution: {main_contribution}" if main_contribution else ""
        prompt = f"""
You are the Summary Agent for an AI Research Paper Assistant.
Create a brief, clear, and easy-to-read summary of the following paper.

Title: {title}
Abstract: {abstract}{contribution_context}

Your summary should be 3 to 4 sentences, focusing on the core problem solved, the methodology proposed, and key results. Write in simple, natural English.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"SummaryAgent Exception: {e}")
            return f"Summary generation failed. The paper discusses {title} and its application to machine learning."
