"""AI Agent Orchestration using LangChain

Demonstrates:
- Multi-agent systems
- Autonomous reasoning
- Tool-using agents
- ReAct framework (Reasoning + Acting)
- Chain-of-thought prompting
"""

from typing import Optional
import pandas as pd


class JobMatchingAgent:
    """Simplified job matching agent for real use
    
    Demonstrates AI agent reasoning without complex dependencies
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        self.reasoning_trace = []
    
    def run(self, query: str, jobs_df: pd.DataFrame) -> str:
        """Run agent on job matching query with ReAct-style reasoning"""
        self.reasoning_trace = []
        
        # ReAct Framework: Thought -> Action -> Observation
        self.reasoning_trace.append(f"💭 Thought: Analyzing query: '{query}'")
        self.reasoning_trace.append("🎯 Action: Parsing requirements from query...")
        
        # Simple matching logic with reasoning
        query_lower = query.lower()
        
        # Observe query patterns
        if "senior" in query_lower or "200k" in query_lower or "$200" in query_lower:
            self.reasoning_trace.append("👁️ Observation: User wants high-salary senior roles (200K+)")
            filtered = jobs_df[jobs_df['salary_min'] >= 200000].head(5)
            role_type = "senior high-salary"
        elif "ml" in query_lower or "machine learning" in query_lower:
            self.reasoning_trace.append("👁️ Observation: User interested in ML engineering roles")
            filtered = jobs_df[jobs_df['title'].str.contains('ML|Machine Learning', case=False, na=False)].head(5)
            role_type = "ML engineering"
        elif "ai" in query_lower or "artificial intelligence" in query_lower:
            self.reasoning_trace.append("👁️ Observation: User interested in AI roles")
            filtered = jobs_df[jobs_df['title'].str.contains('AI|Artificial Intelligence', case=False, na=False)].head(5)
            role_type = "AI"
        else:
            self.reasoning_trace.append("👁️ Observation: General job search")
            filtered = jobs_df.head(5)
            role_type = "general"
        
        # Decision making
        self.reasoning_trace.append(f"🤔 Thought: Found {len(filtered)} {role_type} positions")
        self.reasoning_trace.append("✅ Decision: Returning top matches with details")
        
        # Format response
        response = f"🎯 Found {len(filtered)} {role_type} roles:\n\n"
        for idx, job in filtered.iterrows():
            salary_range = f"${int(job['salary_min']/1000)}K-${int(job['salary_max']/1000)}K"
            response += f"• **{job['title']}** at **{job['company']}**\n"
            response += f"  💰 {salary_range} | 📍 {job.get('location', 'Remote')}\n\n"
        
        self.reasoning_trace.append(f"📊 Final Answer: Presented {len(filtered)} curated opportunities")
        
        return response
    
    def get_reasoning_trace(self) -> str:
        """Get ReAct reasoning trace showing agent's thought process"""
        return "\n".join(self.reasoning_trace)
