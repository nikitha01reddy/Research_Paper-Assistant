import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure the src folder is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search_engine import PaperSearchEngine
from agents.planner_agent import PlannerAgent
from agents.search_agent import SearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.summary_agent import SummaryAgent
from agents.recommendation_agent import RecommendationAgent

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Agentic AI Research Paper Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium look (dark theme enhancements)
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stButton>button {
        background-image: linear-gradient(to right, #4f46e5, #06b6d4);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    .paper-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tag {
        display: inline-block;
        background-color: #312e81;
        color: #c7d2fe;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        border: 1px solid #4338ca;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------- #
# Lazy Resource Caching
# ---------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Initializing Vector Database and Models (first run only)...")
def get_search_engine():
    # Initialize without summarizer/keybert if we default to Gemini,
    # but keep them active as fallbacks.
    return PaperSearchEngine(load_summarizer=True, load_keybert=True)

try:
    engine = get_search_engine()
except Exception as e:
    st.error(f"Failed to load search engine / index. Have you run `python src/build_index.py` first?\n\nError: {e}")
    st.stop()

# Initialize Agents
@st.cache_resource
def get_agents():
    planner = PlannerAgent()
    searcher = SearchAgent(engine)
    analyzer = AnalysisAgent()
    summarizer = SummaryAgent()
    recommender = RecommendationAgent(engine)
    return planner, searcher, analyzer, summarizer, recommender

try:
    planner, searcher, analyzer, summarizer, recommender = get_agents()
    api_key_configured = True
except Exception as e:
    st.warning("GEMINI_API_KEY is not configured or invalid. The system will fall back to local models (BART + KeyBERT).")
    api_key_configured = False

# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.header("🤖 Agentic Assistant Settings")
    top_k = st.slider("Candidate papers to retrieve", min_value=1, max_value=10, value=5)
    st.markdown("---")
    
    st.subheader("Architecture Workflow")
    st.markdown(
        """
        1. **Planner Agent**: Analyzes user query & formulates search queries.
        2. **Search Agent**: Semantic search over 50,000 ArXiv papers (FAISS).
        3. **Analysis Agent**: Extracts structured paper metadata.
        4. **Summary Agent**: Writes concise, readable summary.
        5. **Recommendation Agent**: Recommends similar papers.
        """
    )
    st.markdown("---")
    st.caption("Powered by Sentence-Transformers, FAISS, and Google Gemini.")

# ---------------------------------------------------------------------- #
# Header
# ---------------------------------------------------------------------- #
st.title("🤖 Agentic AI Research Paper Assistant")
st.caption(
    "Analyze & explore 50,000 ArXiv Machine Learning papers using an Agentic AI workflow."
)

query = st.text_input(
    "🔍 What research topic or paper are you looking for today?",
    placeholder="e.g. attention mechanisms for sequence-to-sequence learning",
)

search_clicked = st.button("Query Agent Workflow", type="primary")

# ---------------------------------------------------------------------- #
# Orchestrator Execution
# ---------------------------------------------------------------------- #
if search_clicked and query.strip():
    # Session state initialization for selected paper analysis
    st.session_state.search_performed = True
    st.session_state.query = query
    st.session_state.results = []
    st.session_state.planner_explanation = ""
    st.session_state.planner_queries = []
    
    # 1. Planner Agent
    with st.status("🧠 Planner Agent: Analyzing query and formulating search strategy...", expanded=True) as status:
        if api_key_configured:
            plan = planner.plan(query)
            explanation = plan.get("explanation", "")
            search_queries = plan.get("search_queries", [query])
        else:
            explanation = f"Searching paper database directly for: '{query}' using local heuristics."
            search_queries = [query]
        
        st.write(f"**Search Plan:** {explanation}")
        st.write(f"**Formulated Queries:** `{search_queries}`")
        
        st.session_state.planner_explanation = explanation
        st.session_state.planner_queries = search_queries
        status.update(label="🧠 Planner Agent: Search plan finalized!", state="complete", expanded=False)

    # 2. Search Agent
    with st.status("🔍 Search Agent: Retrieving and merging candidate papers from FAISS index...", expanded=True) as status:
        results = searcher.search_papers(search_queries, k=top_k)
        st.session_state.results = results
        status.update(label=f"🔍 Search Agent: Found {len(results)} relevant candidate papers!", state="complete", expanded=False)

# Display Results
if st.session_state.get("search_performed", False):
    results = st.session_state.get("results", [])
    
    if not results:
        st.warning("No papers found matching the query strategy. Try a different query.")
    else:
        st.success(f"Retrieved {len(results)} candidate papers. Select a paper below to run Analysis, Summary and Recommendation Agents.")

        # Create two main columns: Left for Search Results list, Right for detailed Agent analysis
        left_col, right_col = st.columns([2, 3])

        with left_col:
            st.subheader("📚 Candidate Papers")
            # Let user select which paper to analyze in detail using a radio button list
            paper_titles = [f"{r['rank']}. {r['title']} (Similarity: {r['score']:.2f})" for r in results]
            
            selected_option = st.radio(
                "Choose a paper to analyze:",
                options=paper_titles,
                index=0,
                key="selected_paper_option"
            )
            
            # Map selected option back to the result dictionary
            selected_idx = paper_titles.index(selected_option)
            selected_paper = results[selected_idx]

        with right_col:
            st.subheader("🔬 Active Agent Analysis")
            
            # 3. Paper Analysis Agent
            with st.status("🔬 Analysis Agent: Extracting structured metadata...", expanded=True) as status:
                if api_key_configured:
                    analysis = analyzer.analyze_paper(selected_paper)
                else:
                    # Fallback local extraction
                    keywords = [kw for kw, _ in engine.extract_keywords(selected_paper["abstract"])]
                    analysis = {
                        "title": selected_paper["title"],
                        "authors": "Not specified (Local fallback)",
                        "abstract": selected_paper["abstract"],
                        "keywords": keywords,
                        "main_contribution": "Extracted keywords and abstract using local SpaCy/KeyBERT."
                    }
                status.update(label="🔬 Analysis Agent: Metadata extracted successfully!", state="complete", expanded=False)

            # 4. Summary Agent
            with st.status("📝 Summary Agent: Writing concise summary...", expanded=True) as status:
                if api_key_configured:
                    summary = summarizer.summarize_paper(
                        title=analysis.get("title", selected_paper["title"]),
                        abstract=analysis.get("abstract", selected_paper["abstract"]),
                        main_contribution=analysis.get("main_contribution", "")
                    )
                else:
                    # Fallback local summarizer (BART)
                    summary = engine.summarize(selected_paper["abstract"])
                status.update(label="📝 Summary Agent: Paper summary generated!", state="complete", expanded=False)

            # 5. Recommendation Agent
            with st.status("🤝 Recommendation Agent: Querying vector index for similar papers...", expanded=True) as status:
                recommendations = recommender.recommend_similar_papers(
                    target_title=analysis.get("title", selected_paper["title"]),
                    target_abstract=analysis.get("abstract", selected_paper["abstract"]),
                    num_recommendations=3
                )
                status.update(label="🤝 Recommendation Agent: Recommendations generated!", state="complete", expanded=False)

            # ------------------------------------------------------------------ #
            # Final Response Dashboard
            # ------------------------------------------------------------------ #
            st.markdown("### 📄 Paper Report")
            with st.container(border=True):
                st.markdown(f"### {analysis.get('title')}")
                st.markdown(f"**👤 Authors:** {analysis.get('authors')}")
                
                # Main Contribution Alert
                st.info(f"**🧠 Main Contribution:**\n{analysis.get('main_contribution')}")
                
                # Summary
                st.markdown("#### 📝 Concise AI Summary")
                st.write(summary)
                
                # Keywords
                st.markdown("#### 🏷️ Keywords")
                keywords_html = "".join([f"<span class='tag'>{kw}</span>" for kw in analysis.get("keywords", [])])
                st.markdown(keywords_html, unsafe_allow_html=True)
                
                # Expander for Full Abstract
                with st.expander("📄 View Original Abstract"):
                    st.write(analysis.get("abstract"))

            # Recommendations Dashboard
            st.markdown("### 🤝 Recommended Similar Papers")
            if recommendations:
                for idx, rec in enumerate(recommendations):
                    with st.container(border=True):
                        st.markdown(f"**{idx + 1}. {rec['title']}**")
                        st.caption(f"Semantic Relevance Score: {rec['score']:.2f}")
                        with st.expander("View Abstract"):
                            st.write(rec["abstract"])
            else:
                st.write("No recommendations generated.")

elif search_clicked:
    st.warning("Please enter a research topic or search query.")
else:
    st.info("Enter a query above to start the Agentic AI Research Paper workflow.")
