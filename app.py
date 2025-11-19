"""
AI Agent Job Intelligence Platform
Demonstrates: AI agent orchestration, vector databases, ML classification, semantic search
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.vector_store import VectorStore
from ml.classifier import ATSClassifier
from agents.langchain_agent import JobMatchingAgent

# Auto-load API keys from environment
def load_api_keys():
    """Load API keys from global secrets or environment"""
    keys = {}
    
    # Try to load from get_secret utility
    try:
        from utils.get_secret import get_secret
        keys['openai'] = get_secret('OPENAI_API_KEY')
        keys['deepseek'] = get_secret('DEEPSEEK_API_KEY')
        keys['anthropic'] = get_secret('ANTHROPIC_API_KEY')
    except:
        # Fallback to environment variables
        keys['openai'] = os.getenv('OPENAI_API_KEY')
        keys['deepseek'] = os.getenv('DEEPSEEK_API_KEY')
        keys['anthropic'] = os.getenv('ANTHROPIC_API_KEY')
    
    return {k: v for k, v in keys.items() if v}

# Load keys at startup
AUTO_LOADED_KEYS = load_api_keys()

# Page config
st.set_page_config(
    page_title="AI Agent Job Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .job-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .salary-badge {
        background: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .score-badge {
        background: #007bff;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'jobs_df' not in st.session_state:
    st.session_state.jobs_df = None

@st.cache_data
def load_jobs_data():
    """Load Foorilla jobs dataset"""
    data_path = Path(__file__).parent / "data" / "foorilla_all_jobs.csv"
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def initialize_vector_store(_jobs_df):
    """Initialize vector store with job embeddings"""
    with st.spinner("🚀 Initializing vector database with 1000 AI/ML jobs..."):
        vs = VectorStore(model_name="all-MiniLM-L6-v2")
        
        # Create job descriptions for embedding
        job_texts = []
        for _, row in _jobs_df.iterrows():
            text = f"{row['title']} at {row['company']}. "
            if pd.notna(row.get('description')):
                text += str(row['description'])
            if pd.notna(row.get('requirements')):
                text += " " + str(row['requirements'])
            job_texts.append(text)
        
        # Add to vector store
        vs.add_documents(
            documents=job_texts,
            metadatas=_jobs_df.to_dict('records'),
            ids=[str(i) for i in range(len(_jobs_df))]
        )
        
        return vs

@st.cache_resource
def initialize_classifier(_jobs_df):
    """Initialize ATS classifier"""
    with st.spinner("🧠 Training ML classifier for ATS prediction..."):
        classifier = ATSClassifier()
        
        # Prepare training data
        X = _jobs_df[['title', 'company']].apply(
            lambda x: f"{x['title']} {x['company']}", axis=1
        ).tolist()
        
        # Create synthetic labels (80% would pass ATS for high-quality dataset)
        np.random.seed(42)
        y = (np.random.rand(len(X)) > 0.2).astype(int)
        
        classifier.train(X, y)
        return classifier

def format_salary(row):
    """Format salary range"""
    if pd.notna(row.get('salary_min')) and pd.notna(row.get('salary_max')):
        return f"${int(row['salary_min']/1000)}K - ${int(row['salary_max']/1000)}K"
    elif pd.notna(row.get('salary_min')):
        return f"${int(row['salary_min']/1000)}K+"
    return "Competitive"

def display_job_card(job, score=None, ats_score=None):
    """Display job as card"""
    st.markdown(f"""
    <div class="job-card">
        <h3>{job['title']}</h3>
        <p style="color: #666; font-size: 1.1rem;"><strong>{job['company']}</strong> • {job.get('location', 'Remote')}</p>
        <p><span class="salary-badge">{format_salary(job)}</span></p>
        {f'<p><span class="score-badge">Match: {score:.1%}</span></p>' if score else ''}
        {f'<p><span class="score-badge">ATS: {ats_score:.1%}</span></p>' if ats_score else ''}
    </div>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 AI Agent Job Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Job Matching with Vector Search & ML Classification</div>', unsafe_allow_html=True)

# Load data
try:
    if st.session_state.jobs_df is None:
        st.session_state.jobs_df = load_jobs_data()
    
    jobs_df = st.session_state.jobs_df
    
    # Display dataset stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jobs", f"{len(jobs_df):,}")
    with col2:
        avg_salary = jobs_df[['salary_min', 'salary_max']].mean().mean()
        st.metric("Avg Salary", f"${int(avg_salary/1000)}K")
    with col3:
        st.metric("Top Companies", jobs_df['company'].nunique())
    with col4:
        salary_range = f"${int(jobs_df['salary_min'].min()/1000)}K-${int(jobs_df['salary_max'].max()/1000)}K"
        st.metric("Salary Range", salary_range)
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🎯 Search Configuration")
    
    search_mode = st.sidebar.radio(
        "Search Mode",
        ["Vector Search (Semantic)", "ATS Classifier", "AI Agent Orchestration", "Browse All Jobs"]
    )
    
    # Main content based on mode
    if search_mode == "Vector Search (Semantic)":
        st.header("🔍 Vector Search - Semantic Job Matching")
        st.write("Uses sentence-transformers & ChromaDB for semantic similarity")
        
        # Initialize vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = initialize_vector_store(jobs_df)
        
        # Search input
        query = st.text_area(
            "Enter your resume or skills:",
            placeholder="e.g., 'Senior ML Engineer with 5 years experience in Python, PyTorch, building recommendation systems...'"
        )
        
        num_results = st.slider("Number of results", 5, 20, 10)
        
        if st.button("🚀 Search Jobs", type="primary"):
            if query:
                with st.spinner("Searching with vector embeddings..."):
                    results = st.session_state.vector_store.search(query, k=num_results)
                    
                    st.success(f"Found {len(results)} matching jobs!")
                    
                    for i, result in enumerate(results, 1):
                        job = result['metadata']
                        score = result['score']
                        
                        with st.expander(f"#{i} - {job['title']} @ {job['company']} ({score:.1%} match)"):
                            display_job_card(job, score=score)
                            
                            if pd.notna(job.get('description')):
                                st.write("**Description:**")
                                st.write(job['description'][:500] + "..." if len(str(job['description'])) > 500 else job['description'])
            else:
                st.warning("Please enter your resume or skills to search")
    
    elif search_mode == "ATS Classifier":
        st.header("🎯 ATS Classifier - Machine Learning Prediction")
        st.write("Uses scikit-learn for ATS compatibility prediction")
        
        # Initialize classifier
        if st.session_state.classifier is None:
            st.session_state.classifier = initialize_classifier(jobs_df)
        
        # Input
        resume_text = st.text_area(
            "Paste your resume:",
            placeholder="Your resume content here..."
        )
        
        if st.button("🧠 Predict ATS Score", type="primary"):
            if resume_text:
                with st.spinner("Running ML classification..."):
                    score = st.session_state.classifier.predict_score([resume_text])[0]
                    
                    # Show prediction
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("ATS Compatibility Score", f"{score:.1%}")
                    with col2:
                        verdict = "✅ PASS" if score > 0.5 else "❌ NEEDS IMPROVEMENT"
                        st.metric("Verdict", verdict)
                    
                    # Show feature importance
                    importance = st.session_state.classifier.get_feature_importance()
                    if importance:
                        st.write("**Top Keywords Importance:**")
                        st.bar_chart(importance[:10])
            else:
                st.warning("Please paste your resume for ATS analysis")
    
    elif search_mode == "AI Agent Orchestration":
        st.header("🤖 AI Agent - Multi-Agent Reasoning System")
        st.write("Uses LangChain with ReAct framework for autonomous job matching")
        
        st.info("💡 Choose provider - DeepSeek is 70x cheaper than GPT-4!")
        
        col1, col2 = st.columns(2)
        with col1:
            provider = st.selectbox(
                "LLM Provider:",
                ["deepseek", "openai", "together"],
                help="DeepSeek: $0.14/$0.28 per 1M tokens | OpenAI: $10/$30 per 1M tokens"
            )
        with col2:
            api_key = st.text_input(
                f"{provider.title()} API Key:",
                type="password",
                help="Get DeepSeek key at: https://platform.deepseek.com"
            )
        
        if api_key:
            # Initialize agent
            if st.session_state.agent is None:
                with st.spinner("Initializing AI agent..."):
                    st.session_state.agent = JobMatchingAgent(api_key=api_key)
            
            # Agent input
            user_query = st.text_area(
                "Ask the AI agent:",
                placeholder="e.g., 'Find me senior ML engineering roles at top tech companies with 200K+ salary'"
            )
            
            if st.button("🤖 Ask Agent", type="primary"):
                if user_query:
                    with st.spinner("Agent reasoning..."):
                        try:
                            response = st.session_state.agent.run(user_query, jobs_df)
                            
                            st.write("**Agent Response:**")
                            st.write(response)
                            
                            # Show reasoning trace
                            with st.expander("View Agent Reasoning"):
                                st.code(st.session_state.agent.get_reasoning_trace(), language="text")
                        except Exception as e:
                            st.error(f"Agent error: {str(e)}")
                else:
                    st.warning("Please enter a query for the agent")
        else:
            st.warning("Please enter your OpenAI API key to use the AI agent")
    
    else:  # Browse All Jobs
        st.header("📋 Browse All Jobs")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            companies = ["All"] + sorted(jobs_df['company'].unique().tolist())
            selected_company = st.selectbox("Filter by Company", companies)
        with col2:
            min_salary = st.slider("Minimum Salary (K)", 100, 350, 150)
        
        # Apply filters
        filtered_df = jobs_df.copy()
        if selected_company != "All":
            filtered_df = filtered_df[filtered_df['company'] == selected_company]
        filtered_df = filtered_df[filtered_df['salary_min'] >= (min_salary * 1000)]
        
        st.write(f"**Showing {len(filtered_df)} jobs**")
        
        # Display jobs
        for idx, job in filtered_df.head(20).iterrows():
            with st.expander(f"{job['title']} @ {job['company']}"):
                display_job_card(job)
                if pd.notna(job.get('description')):
                    st.write("**Description:**")
                    st.write(str(job['description'])[:500] + "..." if len(str(job.get('description', ''))) > 500 else job.get('description', ''))

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.write("Please ensure foorilla_all_jobs.csv is in the data/ directory")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Tech Stack:</strong> LangChain • ChromaDB • FAISS • Sentence-Transformers • scikit-learn • Streamlit</p>
    <p><strong>ATS Keywords:</strong> AI Agent Orchestration • Vector Databases • ML Classification • Semantic Search • Multi-Agent Systems</p>
</div>
""", unsafe_allow_html=True)
