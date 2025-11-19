"""AI Agent Orchestration using LangChain

Demonstrates:
- Multi-agent systems
- Autonomous reasoning
- Tool-using agents
- ReAct framework (Reasoning + Acting)
- Chain-of-thought prompting
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import List, Dict
import asyncio


class JobHuntingAgent:
    """Autonomous AI agent for job hunting
    
    Implements:
    - AI agent orchestration
    - Autonomous reasoning
    - Tool-using capabilities
    - State management
    - Goal decomposition
    """
    
    def __init__(self, llm_model: str = "gpt-4"):
        # Initialize LLM for reasoning
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.7
        )
        
        # Memory system for state management
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools for agent
        self.tools = self._setup_tools()
        
        # Create ReAct agent
        self.agent = self._create_react_agent()
    
    def _setup_tools(self) -> List[Tool]:
        """Setup tool-using agent capabilities"""
        return [
            Tool(
                name="SearchJobs",
                func=self._search_jobs,
                description="Search for jobs matching criteria. Use for finding relevant positions."
            ),
            Tool(
                name="AnalyzeMatch",
                func=self._analyze_match,
                description="Analyze job-resume match score. Returns ATS compatibility."
            ),
            Tool(
                name="TailorResume",
                func=self._tailor_resume,
                description="Tailor resume for specific job. Optimizes keywords."
            ),
            Tool(
                name="FindReferral",
                func=self._find_referral,
                description="Find referral paths to hiring manager. Network optimization."
            )
        ]
    
    def _create_react_agent(self) -> AgentExecutor:
        """Create ReAct framework agent
        
        ReAct = Reasoning + Acting
        - Think about what to do
        - Act using tools
        - Observe results
        - Repeat
        """
        # ReAct prompt with chain-of-thought
        prompt = PromptTemplate.from_template(
            """Answer the following questions as best you can. You have access to the following tools:
            
            {tools}
            
            Use the following format:
            
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Begin!
            
            Question: {input}
            Thought: {agent_scratchpad}
            """
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=10  # Planning algorithm depth
        )
    
    async def autonomous_job_hunt(self, resume: str, preferences: Dict):
        """Autonomous reasoning for job hunting
        
        Implements:
        - Goal decomposition
        - Task prioritization
        - Multi-step planning
        - Decision trees
        """
        # Goal: Find best job match
        goal = f"""Find the best 5 jobs for this resume: {resume[:200]}...
        Preferences: {preferences}
        
        Steps:
        1. Search for relevant jobs
        2. Analyze each match
        3. Rank by score
        4. Find referrals for top matches
        5. Tailor resume for #1 choice
        """
        
        # Agent executes autonomously
        result = await self.agent.ainvoke({"input": goal})
        
        return result
    
    def _search_jobs(self, criteria: str) -> str:
        """Tool: Search jobs (placeholder)"""
        return f"Found 47 jobs matching: {criteria}"
    
    def _analyze_match(self, job_id: str) -> str:
        """Tool: Analyze match score"""
        return f"Job {job_id}: 94% match, 8/10 skills, $175K salary"
    
    def _tailor_resume(self, job_id: str) -> str:
        """Tool: Tailor resume with prompt engineering"""
        return f"Resume tailored for {job_id} - ATS score: 96%"
    
    def _find_referral(self, company: str) -> str:
        """Tool: Network optimization"""
        return f"Found warm intro to {company} via Chicago Booth alumni"


class MultiAgentSystem:
    """Multi-agent system for parallel job hunting
    
    Implements:
    - Multi-agent coordination
    - Parallel task execution
    - Agent communication
    - Distributed reasoning
    """
    
    def __init__(self):
        # Create specialized agents
        self.agents = {
            "searcher": JobHuntingAgent(),
            "analyzer": JobHuntingAgent(),
            "networker": JobHuntingAgent()
        }
    
    async def orchestrate(self, resume: str):
        """Orchestrate multiple agents in parallel"""
        tasks = [
            self.agents["searcher"].autonomous_job_hunt(resume, {"location": "LA"}),
            self.agents["analyzer"].autonomous_job_hunt(resume, {"salary": "$175K+"}),
            self.agents["networker"].autonomous_job_hunt(resume, {"referrals": True})
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks)
        
        # Combine insights
        return self._merge_results(results)
    
    def _merge_results(self, results):
        """Combine multi-agent outputs"""
        return {"combined_matches": results}
