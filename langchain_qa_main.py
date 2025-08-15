"""
Question-Driven Multi-Agent Data Analysis System with LangChain Components
Enhanced to accept and answer specific questions about data
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

class QuestionClassifier:
    """Classifies questions to route them to appropriate agents"""
    
    DATA_PROCESSING_KEYWORDS = [
        'clean', 'missing', 'null', 'outlier', 'duplicate', 'quality', 
        'preprocess', 'transform', 'prepare', 'format', 'structure'
    ]
    
    ANALYSIS_KEYWORDS = [
        'correlation', 'pattern', 'trend', 'relationship', 'significant',
        'compare', 'distribution', 'statistics', 'average', 'mean', 'median',
        'hypothesis', 'test', 'insight', 'finding', 'business', 'AUC', 'find', 'biomarker', 'cancer type'
    ]
    
    VISUALIZATION_KEYWORDS = [
        'plot', 'chart', 'graph', 'visualize', 'show', 'display',
        'dashboard', 'histogram', 'scatter', 'bar', 'pie', 'heatmap', 'ROC'
    ]
    
    @staticmethod
    def classify_question(question: str) -> list:
        """Classify question to determine which agents should handle it"""
        question_lower = question.lower()
        agents_needed = []
        
        # Check for data processing needs
        if any(keyword in question_lower for keyword in QuestionClassifier.DATA_PROCESSING_KEYWORDS):
            agents_needed.append('data_scientist')
        
        # Check for analysis needs
        if any(keyword in question_lower for keyword in QuestionClassifier.ANALYSIS_KEYWORDS):
            agents_needed.append('data_analyst')
        
        # Check for visualization needs
        if any(keyword in question_lower for keyword in QuestionClassifier.VISUALIZATION_KEYWORDS):
            agents_needed.append('data_visualizer')
        
        # If no specific keywords found, route to analyst for general questions
        if not agents_needed:
            agents_needed.append('data_analyst')
        
        return agents_needed

class DataInsightsParser(BaseOutputParser):
    """Custom parser for data insights"""
    
    def parse(self, text: str) -> dict:
        """Parse the AI response into structured insights"""
        return {
            "insights": text,
            "timestamp": datetime.now().isoformat()
        }

class QuestionDrivenMultiAgentAnalysis:
    """Multi-agent system that answers questions about data using LangChain components"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Verify API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")
        
        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            max_tokens=1000
        )
        
        # Create output directory
        self.output_dir = "outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize agents
        self.data_scientist = self._create_data_scientist_agent()
        self.data_analyst = self._create_data_analyst_agent() 
        self.data_visualizer = self._create_visualization_agent()
        
        # Store data for question answering
        self.current_data = None
        self.data_summary = None
        
        print("🤖 Biomarker Discover Multi-Agent System Ready!")
        print(f"📊 Using model: {os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')}")
        print(f"🔗 LangChain components: ChatOpenAI, Chains, Custom Parsers")
        print(f"❓ You can ask questions about biomarker discovery and analysis from CancerSeek dataset!")
        print(f" Written by: Hamid Khoshfekr Rudsari, PhD, contact: khoshfekr1994@gmail.com, https://github.com/khoshfekr1994, https://www.linkedin.com/in/hamid-khoshfekr-rudsari-379414b7/")
    
    def _create_data_scientist_agent(self):
        """Create Data Scientist Agent for data processing questions"""
        
        prompt_template = PromptTemplate(
            input_variables=["question", "data_info"],
            template="""You are a Senior Data Scientist with 10+ years of experience.

Your expertise: Data preprocessing, feature engineering, data quality assessment, statistical preprocessing.

Dataset Information:
{data_info}

Question: {question}

Answer the question using your data science expertise. Focus on:
- Data quality and preprocessing aspects
- Technical data handling recommendations
- Feature engineering opportunities
- Data cleaning and transformation strategies
- Converting biomarker columns to numerical values

Provide practical, actionable answers based on the dataset characteristics."""
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            output_parser=StrOutputParser()
        )
        
        return chain
    
    def _create_data_analyst_agent(self):
        """Create Data Analyst Agent for analytical questions"""
        
        prompt_template = PromptTemplate(
            input_variables=["question", "data_info"],
            template="""You are a Senior Data Analyst specializing in statistical analysis and business intelligence.

Your expertise: Statistical methods, hypothesis testing, pattern recognition, business insights.

Dataset Information:
{data_info}

Question: {question}

Answer the question using your analytical expertise. Focus on:
- Statistical patterns and relationships
- Business insights and implications
- Data-driven recommendations
- Quantitative analysis and interpretation
- Finding cancer types and biomarkers
- Finding the AUC of biomarkers for each cancer type vs Normal controls
- For female cancer types, such as breast cancer, ovarian cancer, etc., find the biomarkers that are different between female cancer cases and female controls

Provide detailed analytical insights with statistical evidence where relevant."""
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            output_parser=DataInsightsParser()
        )
        
        return chain
    
    def _create_visualization_agent(self):
        """Create Visualization Specialist Agent for visualization questions"""
        
        prompt_template = PromptTemplate(
            input_variables=["question", "data_info"],
            template="""You are a Data Visualization Specialist expert in visual storytelling and dashboard design.

Your expertise: Visual design principles, chart selection, dashboard creation, data storytelling.

Dataset Information:
{data_info}

Question: {question}

Answer the question using your visualization expertise. Focus on:
- Appropriate chart types and visualization strategies
- Visual storytelling techniques
- Dashboard design recommendations
- Best practices for data presentation
- Creating ROC curves for biomarkers for each cancer type vs Normal controls


If the question asks for a visualization, provide detailed Python code using matplotlib/seaborn.
Make all visualizations professional and publication-ready."""
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            output_parser=DataInsightsParser()
        )
        
        return chain
    
    def load_data(self, data_source):
        """Load and prepare data for question answering"""
        
        if isinstance(data_source, str):
            print(f"📊 Loading data from: {data_source}")
            
            if data_source.lower().endswith(('.xlsx', '.xls')):
                self.current_data = pd.read_excel(data_source)
            elif data_source.lower().endswith('.csv'):
                self.current_data = pd.read_csv(data_source)
            else:
                print("❌ Unsupported format. Use .xlsx, .xls, or .csv")
                return False
        else:
            self.current_data = data_source
        
        print(f"✅ Data loaded! Shape: {self.current_data.shape}")
        
        # Clean data
        print("🧹 Preprocessing data...")
        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.current_data.select_dtypes(include=['object']).columns
        
        # Handle missing values
        for col in numeric_cols:
            if self.current_data[col].isnull().any():
                self.current_data[col] = self.current_data[col].fillna(self.current_data[col].median())
        
        for col in categorical_cols:
            if self.current_data[col].isnull().any() and not self.current_data[col].empty and len(self.current_data[col].mode()) > 0:
                self.current_data[col] = self.current_data[col].fillna(self.current_data[col].mode()[0])
        
        # Generate data summary for agents
        self._generate_data_summary()
        
        print("✅ Data ready for question answering!")
        return True
    
    def _generate_data_summary(self):
        """Generate comprehensive data summary for agents"""
        
        data = self.current_data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Basic statistics
        basic_stats = ""
        if len(numeric_cols) > 0:
            basic_stats = data[numeric_cols].describe().to_string()
        
        # Top categories for categorical columns
        cat_info = ""
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            top_values = data[col].value_counts().head(3)
            cat_info += f"\n{col}: {', '.join([f'{val} ({count})' for val, count in top_values.items()])}"
        
        self.data_summary = f"""
Dataset Shape: {data.shape[0]:,} rows × {data.shape[1]} columns
Columns: {', '.join(data.columns[:15].tolist())}{'... (showing first 15)' if len(data.columns) > 15 else ''}

Numeric Features ({len(numeric_cols)}): {', '.join(numeric_cols[:10].tolist())}{'...' if len(numeric_cols) > 10 else ''}
Categorical Features ({len(categorical_cols)}): {', '.join(categorical_cols[:5].tolist())}{'...' if len(categorical_cols) > 5 else ''}

Missing Values: {data.isnull().sum().sum():,} total
Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Basic Statistics:
{basic_stats}

Top Categories:{cat_info}
        """
    
    def ask_question(self, question: str, create_visualizations: bool = True):
        """Answer a specific question about the loaded data"""
        
        if self.current_data is None:
            return "❌ No data loaded. Please load data first using load_data() method."
        
        print(f"❓ Question: {question}")
        print("🤖 Analyzing question and routing to appropriate agents...")
        
        # Classify question to determine which agents to use
        agents_needed = QuestionClassifier.classify_question(question)
        print(f"📋 Agents assigned: {', '.join(agents_needed)}")
        
        results = {}
        
        # Route question to appropriate agents
        for agent_type in agents_needed:
            print(f"  🤖 {agent_type.replace('_', ' ').title()} working...")
            
            try:
                if agent_type == 'data_scientist':
                    result = self.data_scientist.run(
                        question=question,
                        data_info=self.data_summary
                    )
                    results['data_scientist'] = result
                    
                elif agent_type == 'data_analyst':
                    result = self.data_analyst.run(
                        question=question,
                        data_info=self.data_summary
                    )
                    results['data_analyst'] = result.get('insights', str(result)) if isinstance(result, dict) else result
                    
                elif agent_type == 'data_visualizer':
                    result = self.data_visualizer.run(
                        question=question,
                        data_info=self.data_summary
                    )
                    results['data_visualizer'] = result.get('insights', str(result)) if isinstance(result, dict) else result
                    
                    # Execute visualization code if present and requested
                    if create_visualizations and 'plt.' in results['data_visualizer']:
                        self._execute_visualization_code(results['data_visualizer'], question)
                        
            except Exception as e:
                results[agent_type] = f"Error: {str(e)}"
        
        # Generate comprehensive answer
        answer = self._synthesize_answers(question, results)
        
        # Save the Q&A to a log file
        self._log_question_answer(question, answer)
        
        return answer
    
    def _execute_visualization_code(self, viz_response: str, question: str):
        """Execute visualization code from the response"""
        
        try:
            # Extract Python code from the response
            code_pattern = r'```python\n(.*?)\n```'
            code_matches = re.findall(code_pattern, viz_response, re.DOTALL)
            
            if code_matches:
                print("  🎨 Creating visualization...")
                
                # Prepare safe execution environment
                safe_globals = {
                    'pd': pd,
                    'np': np,
                    'plt': plt,
                    'sns': sns,
                    'data': self.current_data,
                    'df': self.current_data,
                    'os': os
                }
                
                # Execute the visualization code
                exec(code_matches[0], safe_globals)
                
                # Save the plot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"question_viz_{timestamp}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"  ✅ Visualization saved: {filename}")
                
        except Exception as e:
            print(f"  ⚠️ Could not execute visualization code: {e}")
    
    def _synthesize_answers(self, question: str, results: dict) -> str:
        """Synthesize answers from multiple agents into a coherent response"""
        
        answer = f"# Answer to: {question}\n\n"
        
        if 'data_scientist' in results:
            answer += "## 🔬 Data Science Perspective\n"
            answer += f"{results['data_scientist']}\n\n"
        
        if 'data_analyst' in results:
            answer += "## 📈 Analytical Insights\n"
            answer += f"{results['data_analyst']}\n\n"
        
        if 'data_visualizer' in results:
            answer += "## 🎨 Visualization Recommendations\n"
            answer += f"{results['data_visualizer']}\n\n"
        
        answer += "---\n*Answer generated by Multi-Agent Analysis System*"
        
        return answer
    
    def _log_question_answer(self, question: str, answer: str):
        """Log question and answer to a file"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"""
## [{timestamp}] Question: {question}

{answer}

{'='*80}
"""
        
        log_file = os.path.join(self.output_dir, 'question_log.md')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def interactive_mode(self):
        """Start interactive question-answering mode"""
        
        if self.current_data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*60)
        print("🤖 INTERACTIVE QUESTION MODE")
        print("="*60)
        print("Ask questions about your data! Type 'quit' to exit.")
        print("Examples:")
        print("  - What are the missing values in this dataset?")
        print("  - Show me a correlation heatmap")
        print("  - What are the main trends in the data?")
        print("  - Create a histogram of the sales column")
        print()
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n" + "-"*50)
                answer = self.ask_question(question)
                print("\n📝 Answer:")
                print(answer)
                print("-"*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def auto_analysis_mode(self, analysis_objectives="general analysis"):
        """Perform automatic comprehensive analysis (original functionality)"""
        
        if self.current_data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("🚀 Running automatic comprehensive analysis...")
        
        # Generate automatic questions based on data characteristics
        auto_questions = self._generate_auto_questions()
        
        results = {}
        for question in auto_questions:
            print(f"\n🤖 Auto-question: {question}")
            answer = self.ask_question(question, create_visualizations=True)
            results[question] = answer
        
        # Generate comprehensive report
        self._generate_auto_analysis_report(results)
        
        print("✅ Comprehensive analysis completed!")
        return results
    
    def _generate_auto_questions(self):
        """Generate automatic questions based on data characteristics"""
        
        questions = [
            "What is the overall quality of this dataset and what preprocessing is needed?",
            "What are the key statistical patterns and relationships in the data?",
            "Create appropriate visualizations to show the main insights in this dataset",
            "What are the most important findings and business recommendations from this data?"
        ]
        
        # Add specific questions based on data characteristics
        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.current_data.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 1:
            questions.append("Show me the correlation patterns between numeric variables")
        
        if len(categorical_cols) > 0:
            questions.append(f"Analyze the distribution of categorical variables, especially {categorical_cols[0]}")
        
        if self.current_data.isnull().sum().sum() > 0:
            questions.append("How should I handle the missing values in this dataset?")
        
        return questions
    
    def _generate_auto_analysis_report(self, results: dict):
        """Generate comprehensive auto-analysis report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# 🤖 Comprehensive Data Analysis Report

**Generated on:** {timestamp}  
**Analysis Type:** Question-Driven Multi-Agent Analysis  
**Dataset:** {self.current_data.shape[0]:,} rows × {self.current_data.shape[1]} columns

---

## 📊 Executive Summary

This report presents a comprehensive analysis of your dataset using our multi-agent question-answering system. Each section below represents insights from specialized AI agents answering key questions about your data.

---

"""
        
        for i, (question, answer) in enumerate(results.items(), 1):
            report += f"## Question {i}: {question}\n\n{answer}\n\n---\n\n"
        
        report += f"""
## 🎯 Analysis Summary

- **Questions Analyzed:** {len(results)}
- **Agents Consulted:** Data Scientist, Data Analyst, Visualization Specialist
- **Visualizations Created:** Check the outputs folder for generated charts
- **Data Quality:** {'Good' if self.current_data.isnull().sum().sum() == 0 else 'Needs attention'}

---

*Generated by Question-Driven Multi-Agent Analysis System*
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'comprehensive_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  ✅ Comprehensive report saved: {report_path}")

def main():
    """Main execution function"""
    print("🚀 Question-Driven Multi-Agent Data Analysis System")
    print("=" * 65)
    print("🔗 Powered by LangChain + OpenAI")
    print("🤖 3 Specialized AI Agents ready to answer your questions")
    print("❓ Interactive question-answering mode available")
    print()
    
    try:
        # Initialize system
        analyzer = QuestionDrivenMultiAgentAnalysis()
    except ValueError as e:
        print(f"❌ {e}")
        print("\n🔑 Setup required:")
        print("1. Create .env file with: OPENAI_API_KEY=your_key_here")
        print("2. Add credits to your OpenAI account")
        print("3. Install dependencies: pip install -r simple_requirements.txt")
        return
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n📦 Install missing dependencies:")
        print("pip install -r simple_requirements.txt")
        return
    
    # Find data file
    data_file = None
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        if not os.path.exists(data_file):
            print(f"❌ File not found: {data_file}")
            data_file = None
    
    if not data_file:
        # Look for common files
        common_files = ['cancer_seek_protein_2.xlsx', 'data.xlsx', 'data.csv']
        for file in common_files:
            if os.path.exists(file):
                data_file = file
                print(f"📁 Found data file: {data_file}")
                break
    
    if not data_file:
        print("❌ No data file found")
        print("Usage: python langchain_qa_main.py your_data_file.xlsx")
        print("Or place a data file (xlsx/csv) in the current directory")
        return
    
    # Load data
    if not analyzer.load_data(data_file):
        return
    
    print("\n" + "="*65)
    print("🎯 CHOOSE YOUR MODE:")
    print("="*65)
    print("1. Interactive Mode - Ask specific questions about your data")
    print("2. Auto Analysis Mode - Comprehensive automatic analysis")
    print("3. Single Question Mode - Ask one question and exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            analyzer.interactive_mode()
        elif choice == '2':
            analyzer.auto_analysis_mode()
        elif choice == '3':
            question = input("\n❓ What would you like to know about your data? ")
            if question.strip():
                answer = analyzer.ask_question(question)
                print("\n📝 Answer:")
                print(answer)
        else:
            print("Invalid choice. Starting interactive mode...")
            analyzer.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()