"""
Multi-Agent System for Data Analysis
Defines specialized agents for data science tasks
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

class SimpleAgent:
    """Simple agent implementation using OpenAI API"""
    
    def __init__(self, role, goal, backstory):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.client = client
        self.model = model_name
    
    def execute_task(self, task_description):
        """Execute a task using the agent's expertise"""
        system_prompt = f"""
        You are a {self.role}.
        
        Goal: {self.goal}
        
        Background: {self.backstory}
        
        Please approach this task with your specialized expertise and provide detailed, actionable results.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_description}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error executing task: {str(e)}"

class DataAgents:
    """Factory class for creating specialized data agents"""
    
    @staticmethod
    def data_scientist_agent():
        """Agent specialized in data science tasks like cleaning, preprocessing, and feature engineering"""
        return SimpleAgent(
            role="Senior Data Scientist",
            goal="Process, clean, and prepare data for analysis while identifying key patterns and insights",
            backstory="""You are a seasoned data scientist with 10+ years of experience in data preprocessing, 
            feature engineering, and exploratory data analysis. You excel at identifying data quality issues, 
            handling missing values, and transforming raw data into analysis-ready formats. You always consider 
            statistical assumptions and data distributions in your work."""
        )
    
    @staticmethod
    def data_analyst_agent():
        """Agent specialized in statistical analysis and hypothesis testing"""
        return SimpleAgent(
            role="Senior Data Analyst",
            goal="Perform comprehensive statistical analysis and extract meaningful insights from processed data",
            backstory="""You are an expert data analyst with deep knowledge of statistical methods, hypothesis testing,
            and business intelligence. You specialize in uncovering trends, correlations, and patterns that drive 
            business decisions. You're proficient in various analytical techniques from descriptive statistics to 
            advanced modeling approaches."""
        )
    
    @staticmethod
    def data_visualization_agent():
        """Agent specialized in creating compelling visualizations and dashboards"""
        return SimpleAgent(
            role="Data Visualization Specialist",
            goal="Create clear, informative, and visually appealing charts and dashboards that communicate insights effectively",
            backstory="""You are a data visualization expert who transforms complex analytical results into clear, 
            actionable visual stories. You understand principles of visual design, color theory, and effective 
            communication through charts. You're skilled in various visualization libraries and always consider 
            the target audience when designing visualizations."""
        )
    
    @staticmethod
    def get_all_agents():
        """Return all agents as a dictionary"""
        return {
            "data_scientist": DataAgents.data_scientist_agent(),
            "data_analyst": DataAgents.data_analyst_agent(),
            "data_visualizer": DataAgents.data_visualization_agent()
        }