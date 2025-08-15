# 🧬 Multi-Agentic Biomarker Analyzer

> **AI-Powered Biomarker Data Analysis using LangChain Multi-Agent Architecture**

A sophisticated multi-agent system that leverages LangChain and OpenAI to perform comprehensive biomarker data analysis. The system employs three specialized AI agents to handle data preprocessing, statistical analysis, and visualization generation for biomarker research and discovery.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0+-green.svg)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🔬 Overview

The Multi-Agentic Biomarker Analyzer is designed specifically for biomarker research, protein analysis, and clinical data interpretation. It combines the power of Large Language Models with specialized data science techniques to provide comprehensive insights into biomarker datasets.

### 🎯 Key Features

- **🤖 Three Specialized AI Agents**:
  - **Data Scientist Agent**: Data preprocessing, quality assessment, feature engineering
  - **Data Analyst Agent**: Statistical analysis, hypothesis testing, pattern recognition
  - **Visualization Specialist Agent**: Professional charts, dashboards, visual storytelling

- **🧬 Biomarker-Focused Analysis**:
  - Protein expression analysis
  - Clinical correlation studies
  - Biomarker discovery workflows
  - Statistical validation of findings

- **📊 Comprehensive Visualizations**:
  - Executive dashboards for stakeholders
  - Correlation matrices for biomarker relationships
  - Distribution analysis for expression levels
  - Outlier detection for quality control

- **🔗 LangChain Integration**:
  - Structured agent workflows
  - Custom output parsers
  - Prompt engineering for domain expertise
  - Scalable multi-agent architecture

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Input Data Layer                    │
│           (Excel, CSV, Clinical Data)               │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              LangChain Orchestration                │
│    ┌─────────────┬─────────────┬─────────────┐      │
│    │Data Scientist│Data Analyst │Visualization│      │
│    │   Agent     │   Agent     │ Specialist  │      │
│    │             │             │   Agent     │      │
│    └─────────────┴─────────────┴─────────────┘      │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│                Output Layer                         │
│    ├── Reports (Markdown)                          │
│    ├── Visualizations (PNG)                        │
│    ├── Processed Data (CSV)                        │
│    └── Executive Dashboards                        │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key with GPT-4 access
- 8GB+ RAM recommended for large datasets

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/khoshfekr1994/multi-agentic-biomarker-analyzer.git
cd multi-agentic-biomarker-analyzer
```

2. **Install dependencies**:
```bash
pip install -r simple_requirements.txt
```

3. **Set up environment variables**:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "OPENAI_MODEL_NAME=gpt-4o-mini" >> .env
```

4. **Run the analyzer**:
```bash
# With your own data file
python langchain_main.py your_biomarker_data.xlsx

# Or use the included sample
python langchain_main.py cancer_seek_protein_2.xlsx
```

## 📁 Project Structure

```
multi-agentic-biomarker-analyzer/
├── langchain_main.py           # Main analysis engine with LangChain
├── agents.py                   # AI agent definitions
├── tasks.py                    # Task orchestration
├── utils.py                    # Utility functions
├── simple_requirements.txt     # Dependencies
├── cancer_seek_protein_2.xlsx  # Sample biomarker dataset
├── .env.example               # Environment variables template
├── outputs/                   # Generated results directory
│   ├── executive_dashboard.png
│   ├── correlation_analysis.png
│   ├── feature_distributions.png
│   ├── outlier_analysis.png
│   ├── langchain_analysis_report.md
│   └── cleaned_dataset.csv
└── README.md                  # This file
```

## 🧪 Usage Examples

### Basic Biomarker Analysis

```python
from langchain_main import LangChainMultiAgentAnalysis

# Initialize the system
analyzer = LangChainMultiAgentAnalysis()

# Analyze biomarker data
results = analyzer.analyze_data_with_agents('protein_expression_data.xlsx')

# Access results
print(f"Generated {len(results['visualizations'])} visualizations")
print(f"Data Scientist insights: {results['data_scientist_insights']}")
```

### Custom Analysis Workflow

```python
from agents import DataAgents
from tasks import DataTasks

# Create specialized agents
agents = DataAgents.get_all_agents()

# Define custom tasks
tasks = DataTasks.create_task_sequence(
    agents=agents,
    data_description="Protein biomarker expression levels",
    analysis_objectives="Identify diagnostic biomarkers",
    visualization_requirements="Clinical presentation charts"
)

# Execute workflow
for task in tasks:
    result = task.execute()
    print(f"Task completed: {result}")
```

## 📊 Generated Outputs

The system automatically generates:

1. **📋 Executive Dashboard** - High-level overview for stakeholders
2. **🔗 Correlation Analysis** - Biomarker relationship matrices
3. **📈 Feature Distributions** - Expression level distributions
4. **🎯 Outlier Detection** - Quality control analysis
5. **📝 Comprehensive Report** - Detailed findings in Markdown
6. **🗃️ Cleaned Dataset** - Processed data for further analysis

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini  # or gpt-4 for enhanced analysis
```

### Supported Data Formats

- **Excel**: `.xlsx`, `.xls`
- **CSV**: `.csv`
- **Requirements**: 
  - Column headers in first row
  - Numeric data for statistical analysis
  - Missing values handled automatically

## 💰 Cost Estimation

- **GPT-4o-mini**: ~$0.10-$0.30 per analysis
- **GPT-4**: ~$0.50-$1.50 per analysis
- Costs depend on dataset size and complexity

## 🛠️ Advanced Features

### Custom Agent Configuration

```python
# Create specialized biomarker agent
biomarker_agent = SimpleAgent(
    role="Biomarker Research Specialist",
    goal="Identify clinically relevant biomarkers",
    backstory="Expert in proteomics and clinical validation"
)
```

### Extensible Visualization

```python
# Add custom visualization types
def create_biomarker_heatmap(data):
    # Custom implementation
    pass
```

## 📄 License

This project is licensed under the MIT License.

- **Email**: [Contact](mailto:khoshfekr1994@example.com)

## Author
- Hamid Khoshfekr Rudsari
- khoshfekr1994@gmail.com
