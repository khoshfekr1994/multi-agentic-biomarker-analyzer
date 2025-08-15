"""
Task definitions for the multi-agent data analysis system
"""

from typing import Dict, Any

class SimpleTask:
    """Simple task implementation"""
    
    def __init__(self, description, agent, expected_output=""):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.context = []
        self.result = None
    
    def execute(self, context_results=None):
        """Execute the task with the assigned agent"""
        task_with_context = self.description
        
        if context_results:
            context_info = "\n\nContext from previous tasks:\n"
            for i, result in enumerate(context_results):
                context_info += f"\n--- Previous Task {i+1} Results ---\n{result}\n"
            task_with_context += context_info
        
        self.result = self.agent.execute_task(task_with_context)
        return self.result

class DataTasks:
    """Factory class for creating data analysis tasks"""
    
    @staticmethod
    def data_processing_task(agent, data_description: str):
        """Task for data scientist to process and clean data"""
        return SimpleTask(
            description=f"""
            Analyze and process the following data: {data_description}
            
            Your responsibilities include:
            1. Load and examine the data structure
            2. Identify and handle missing values
            3. Detect and address outliers
            4. Perform data type conversions if needed
            5. Create new features if beneficial
            6. Validate data quality and consistency
            7. Prepare the data for analysis
            
            Provide a detailed report of all preprocessing steps taken and 
            save the cleaned data for the next agent.
            """,
            agent=agent,
            expected_output="A comprehensive data processing report with cleaned dataset ready for analysis"
        )
    
    @staticmethod
    def data_analysis_task(agent, analysis_objectives: str):
        """Task for data analyst to perform statistical analysis"""
        return SimpleTask(
            description=f"""
            Perform comprehensive statistical analysis on the processed data with focus on: {analysis_objectives}
            
            Your responsibilities include:
            1. Conduct exploratory data analysis (EDA)
            2. Calculate descriptive statistics
            3. Identify correlations and relationships
            4. Perform hypothesis testing where appropriate
            5. Apply relevant statistical models
            6. Validate findings and check assumptions
            7. Summarize key insights and patterns
            
            Provide detailed analytical results with statistical evidence and 
            business recommendations based on your findings.
            """,
            agent=agent,
            expected_output="Detailed statistical analysis report with key insights and recommendations"
        )
    
    @staticmethod
    def data_visualization_task(agent, visualization_requirements: str):
        """Task for visualization agent to create charts and dashboards"""
        return SimpleTask(
            description=f"""
            Create compelling visualizations based on the analysis results with focus on: {visualization_requirements}
            
            Your responsibilities include:
            1. Design appropriate chart types for different data insights
            2. Create clear and informative visualizations
            3. Ensure visual consistency and professional appearance
            4. Add proper titles, labels, and legends
            5. Use appropriate color schemes and styling
            6. Create both summary and detailed views
            7. Generate a dashboard if multiple charts are created
            
            Provide Python code for all visualizations and save the charts as files.
            All visualizations should tell a clear story and be suitable for presentation.
            """,
            agent=agent,
            expected_output="Complete set of publication-ready visualizations with Python code"
        )
    
    @staticmethod
    def create_task_sequence(agents: Dict[str, Any], data_description: str, 
                           analysis_objectives: str, visualization_requirements: str):
        """Create a sequence of tasks for the complete data analysis workflow"""
        
        # Task 1: Data Processing
        processing_task = DataTasks.data_processing_task(
            agents["data_scientist"], 
            data_description
        )
        
        # Task 2: Data Analysis  
        analysis_task = DataTasks.data_analysis_task(
            agents["data_analyst"],
            analysis_objectives
        )
        
        # Task 3: Data Visualization
        visualization_task = DataTasks.data_visualization_task(
            agents["data_visualizer"],
            visualization_requirements
        )
        
        # Return tasks in sequence
        return [processing_task, analysis_task, visualization_task]