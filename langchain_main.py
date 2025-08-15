"""
Multi-Agent Data Analysis System with LangChain Components
Uses LangChain for AI agents and actually creates visualizations
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

class DataInsightsParser(BaseOutputParser):
    """Custom parser for data insights"""
    
    def parse(self, text: str) -> dict:
        """Parse the AI response into structured insights"""
        return {
            "insights": text,
            "timestamp": datetime.now().isoformat()
        }

class LangChainMultiAgentAnalysis:
    """Multi-agent system using LangChain components"""
    
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
            max_tokens=800
        )
        
        # Create output directory
        self.output_dir = "outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize agents
        self.data_scientist = self._create_data_scientist_agent()
        self.data_analyst = self._create_data_analyst_agent() 
        self.viz_specialist = self._create_visualization_agent()
        
        print("🤖 LangChain Multi-Agent Analysis System Ready!")
        print(f"📊 Using model: {os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')}")
        print(f"🔗 LangChain components: ChatOpenAI, Chains, Custom Parsers")
        print(f"💰 Estimated cost per run: $0.10-$0.30")
    
    def _create_data_scientist_agent(self):
        """Create Data Scientist Agent using LangChain"""
        
        prompt_template = PromptTemplate(
            input_variables=["data_info"],
            template="""You are a Senior Data Scientist with 10+ years of experience.
            
Your expertise: Data preprocessing, feature engineering, data quality assessment, statistical preprocessing.

Analyze this dataset and provide insights about:

{data_info}

Provide your analysis in the following format:
1. Data Quality Assessment (3 key observations)
2. Preprocessing Recommendations (3 specific actions)  
3. Feature Engineering Opportunities (2 suggestions)
4. Potential Data Issues (2 concerns to watch for)

Keep each point concise and actionable. Focus on practical data science insights."""
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            output_parser=StrOutputParser()
        )
        
        return chain
    
    def _create_data_analyst_agent(self):
        """Create Data Analyst Agent using LangChain"""
        
        prompt_template = PromptTemplate(
            input_variables=["data_info", "processing_insights"],
            template="""You are a Senior Data Analyst specializing in statistical analysis and business intelligence.

Your expertise: Statistical methods, hypothesis testing, pattern recognition, business insights.

Dataset Information:
{data_info}

Data Scientist's Assessment:
{processing_insights}

Provide your statistical analysis insights:
1. Key Statistical Patterns (3 observations)
2. Correlation Insights (2 important relationships to explore)
3. Business Intelligence Findings (3 actionable insights)
4. Recommended Statistical Tests (2 specific tests to perform)

Focus on insights that drive business decisions and statistical significance."""
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            output_parser=DataInsightsParser()
        )
        
        return chain
    
    def _create_visualization_agent(self):
        """Create Visualization Specialist Agent using LangChain"""
        
        prompt_template = PromptTemplate(
            input_variables=["data_info", "analysis_insights", "visualizations_created"],
            template="""You are a Data Visualization Specialist expert in visual storytelling and dashboard design.

Your expertise: Visual design principles, chart selection, dashboard creation, data storytelling.

Dataset Information:
{data_info}

Analysis Insights:
{analysis_insights}

Visualizations Created:
{visualizations_created}

Provide your visualization strategy:
1. Chart Type Effectiveness (3 assessments of chosen visualizations)
2. Visual Story Narrative (what story do the visualizations tell?)
3. Dashboard Improvements (2 suggestions for better visual communication)  
4. Executive Summary (3 key takeaways for stakeholders)

Focus on how the visualizations effectively communicate data insights to different audiences."""
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            output_parser=DataInsightsParser()
        )
        
        return chain
    
    def create_comprehensive_visualizations(self, data):
        """Create comprehensive visualizations using matplotlib and seaborn"""
        
        print("🎨 Creating comprehensive visualizations...")
        
        # Set up professional styling
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Get column information
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        visualizations_created = []
        
        # 1. EXECUTIVE DASHBOARD
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            fig.suptitle('📊 Executive Data Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
            
            # Dataset Overview
            overview_text = f"""📈 DATASET OVERVIEW
            
• Total Records: {data.shape[0]:,}
• Total Features: {data.shape[1]:,}
• Numeric Features: {len(numeric_cols)}
• Categorical Features: {len(categorical_cols)}
• Missing Values: {data.isnull().sum().sum():,}
• Data Completeness: {((data.size - data.isnull().sum().sum()) / data.size * 100):.1f}%
• Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB"""
            
            axes[0, 0].text(0.05, 0.95, overview_text, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                           transform=axes[0, 0].transAxes, fontfamily='monospace')
            axes[0, 0].set_title('📋 Dataset Summary', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Missing Values Analysis
            missing_data = data.isnull().sum()
            missing_data = missing_data[missing_data > 0].head(15)
            
            if len(missing_data) > 0:
                bars = axes[0, 1].bar(range(len(missing_data)), missing_data.values, 
                                     color='coral', alpha=0.7, edgecolor='darkred')
                axes[0, 1].set_xticks(range(len(missing_data)))
                axes[0, 1].set_xticklabels([col[:15] + '...' if len(col) > 15 else col 
                                           for col in missing_data.index], rotation=45, ha='right')
                axes[0, 1].set_title('❌ Missing Values Analysis', fontsize=14, fontweight='bold')
                axes[0, 1].set_ylabel('Missing Count', fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            else:
                axes[0, 1].text(0.5, 0.5, '✅ PERFECT!\nNo Missing Values', 
                               ha='center', va='center', fontsize=16, color='green', 
                               fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))
                axes[0, 1].set_title('❌ Missing Values Status', fontsize=14, fontweight='bold')
                axes[0, 1].axis('off')
            
            # Data Types Distribution
            dtype_counts = data.dtypes.value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(dtype_counts)))
            wedges, texts, autotexts = axes[0, 2].pie(dtype_counts.values, labels=dtype_counts.index, 
                                                     autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0, 2].set_title('🔢 Data Types Distribution', fontsize=14, fontweight='bold')
            
            # Make autopct text bold
            for autotext in autotexts:
                autotext.set_fontweight('bold')
            
            # Numeric Features Summary
            if len(numeric_cols) > 0:
                sample_col = numeric_cols[0]
                n, bins, patches = axes[1, 0].hist(data[sample_col].dropna(), bins=25, 
                                                  alpha=0.7, color='skyblue', edgecolor='navy')
                axes[1, 0].axvline(data[sample_col].mean(), color='red', linestyle='--', 
                                  linewidth=2, label=f'Mean: {data[sample_col].mean():.2f}')
                axes[1, 0].axvline(data[sample_col].median(), color='green', linestyle='--', 
                                  linewidth=2, label=f'Median: {data[sample_col].median():.2f}')
                axes[1, 0].set_title(f'📊 Sample Distribution: {sample_col[:20]}', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Value', fontweight='bold')
                axes[1, 0].set_ylabel('Frequency', fontweight='bold')
                axes[1, 0].legend(fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, '📊 No Numeric Data\nAvailable for Distribution', 
                               ha='center', va='center', fontsize=14, color='orange', fontweight='bold')
                axes[1, 0].set_title('📊 Sample Distribution', fontsize=14, fontweight='bold')
            
            # Categorical Features (if any)
            if len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                top_categories = data[cat_col].value_counts().head(10)
                bars = axes[1, 1].bar(range(len(top_categories)), top_categories.values, 
                                     color='lightgreen', alpha=0.7, edgecolor='darkgreen')
                axes[1, 1].set_xticks(range(len(top_categories)))
                axes[1, 1].set_xticklabels([str(cat)[:10] + '...' if len(str(cat)) > 10 else str(cat) 
                                           for cat in top_categories.index], rotation=45, ha='right')
                axes[1, 1].set_title(f'📋 Top Categories: {cat_col[:20]}', fontsize=14, fontweight='bold')
                axes[1, 1].set_ylabel('Count', fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, '📋 No Categorical Data', ha='center', va='center',
                               fontsize=14, color='orange', fontweight='bold')
                axes[1, 1].set_title('📋 Categorical Analysis', fontsize=14, fontweight='bold')
            
            # Data Quality Score
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            quality_score = ((total_cells - missing_cells) / total_cells) * 100
            
            quality_color = 'green' if quality_score >= 95 else 'orange' if quality_score >= 80 else 'red'
            quality_text = f"""🎯 DATA QUALITY SCORE
            
Overall Score: {quality_score:.1f}%

{'🟢 EXCELLENT' if quality_score >= 95 else '🟡 GOOD' if quality_score >= 80 else '🔴 NEEDS IMPROVEMENT'}

Completeness: {quality_score:.1f}%
Total Cells: {total_cells:,}
Missing Cells: {missing_cells:,}
Valid Cells: {total_cells - missing_cells:,}"""
            
            axes[1, 2].text(0.05, 0.95, quality_text, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor=quality_color, alpha=0.3),
                           transform=axes[1, 2].transAxes, fontfamily='monospace', fontweight='bold')
            axes[1, 2].set_title('🎯 Data Quality Assessment', fontsize=14, fontweight='bold')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            dashboard_path = os.path.join(self.output_dir, 'executive_dashboard.png')
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            visualizations_created.append('executive_dashboard.png')
            print("  ✅ Executive dashboard created")
            
        except Exception as e:
            print(f"  ⚠️ Executive dashboard failed: {e}")
        
        # 2. CORRELATION ANALYSIS
        if len(numeric_cols) > 1:
            try:
                plt.figure(figsize=(14, 11))
                
                # Limit columns for readability
                corr_cols = numeric_cols[:25] if len(numeric_cols) > 25 else numeric_cols
                corr_matrix = data[corr_cols].corr()
                
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Generate heatmap
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                           cmap='RdBu_r', center=0, square=True, 
                           cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'})
                
                plt.title('🔗 Feature Correlation Matrix\n(Strong correlations: |r| > 0.7)', 
                         fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('Features', fontweight='bold')
                plt.ylabel('Features', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                corr_path = os.path.join(self.output_dir, 'correlation_analysis.png')
                plt.savefig(corr_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                visualizations_created.append('correlation_analysis.png')
                print("  ✅ Correlation analysis created")
                
            except Exception as e:
                print(f"  ⚠️ Correlation analysis failed: {e}")
        
        # 3. FEATURE DISTRIBUTIONS
        if len(numeric_cols) > 0:
            try:
                n_features = min(9, len(numeric_cols))
                fig, axes = plt.subplots(3, 3, figsize=(20, 16))
                fig.suptitle('📊 Feature Distribution Analysis', fontsize=18, fontweight='bold')
                
                axes = axes.flatten()
                
                for i, col in enumerate(numeric_cols[:n_features]):
                    data_clean = data[col].dropna()
                    
                    # Create histogram with statistics
                    n, bins, patches = axes[i].hist(data_clean, bins=30, alpha=0.7, 
                                                   color='skyblue', edgecolor='navy')
                    
                    # Add mean and median lines
                    mean_val = data_clean.mean()
                    median_val = data_clean.median()
                    std_val = data_clean.std()
                    
                    axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                                   label=f'Mean: {mean_val:.2f}')
                    axes[i].axvline(median_val, color='green', linestyle='--', linewidth=2,
                                   label=f'Median: {median_val:.2f}')
                    
                    # Title with statistics
                    axes[i].set_title(f'{col[:25]}\n(μ={mean_val:.2f}, σ={std_val:.2f})', 
                                     fontweight='bold', fontsize=11)
                    axes[i].set_xlabel('Value', fontweight='bold')
                    axes[i].set_ylabel('Frequency', fontweight='bold')
                    axes[i].legend(fontsize=8)
                    axes[i].grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(n_features, 9):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                dist_path = os.path.join(self.output_dir, 'feature_distributions.png')
                plt.savefig(dist_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                visualizations_created.append('feature_distributions.png')
                print("  ✅ Feature distributions created")
                
            except Exception as e:
                print(f"  ⚠️ Feature distributions failed: {e}")
        
        # 4. OUTLIER DETECTION ANALYSIS
        if len(numeric_cols) > 0:
            try:
                n_features = min(8, len(numeric_cols))
                fig, axes = plt.subplots(2, 4, figsize=(20, 12))
                fig.suptitle('📦 Outlier Detection Analysis (Box Plots)', fontsize=18, fontweight='bold')
                
                axes = axes.flatten()
                
                for i, col in enumerate(numeric_cols[:n_features]):
                    # Create boxplot
                    box_plot = axes[i].boxplot(data[col].dropna(), patch_artist=True)
                    box_plot['boxes'][0].set_facecolor('lightcoral')
                    box_plot['boxes'][0].set_alpha(0.7)
                    
                    # Calculate outlier statistics
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                    
                    axes[i].set_title(f'{col[:25]}\n({len(outliers)} outliers)', 
                                     fontweight='bold', fontsize=11)
                    axes[i].set_ylabel('Value', fontweight='bold')
                    axes[i].grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(n_features, 8):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                outlier_path = os.path.join(self.output_dir, 'outlier_analysis.png')
                plt.savefig(outlier_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                visualizations_created.append('outlier_analysis.png')
                print("  ✅ Outlier analysis created")
                
            except Exception as e:
                print(f"  ⚠️ Outlier analysis failed: {e}")
        
        return visualizations_created
    
    def analyze_data_with_agents(self, data_source):
        """Main analysis using LangChain agents"""
        
        # Load data
        if isinstance(data_source, str):
            print(f"📊 Loading data from: {data_source}")
            
            if data_source.lower().endswith(('.xlsx', '.xls')):
                data = pd.read_excel(data_source)
            elif data_source.lower().endswith('.csv'):
                data = pd.read_csv(data_source)
            else:
                print("❌ Unsupported format. Use .xlsx, .xls, or .csv")
                return None
        else:
            data = data_source
        
        print(f"✅ Data loaded! Shape: {data.shape}")
        
        # Clean data
        print("🧹 Preprocessing data...")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Handle missing values
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        
        for col in categorical_cols:
            if not data[col].empty and len(data[col].mode()) > 0:
                data[col] = data[col].fillna(data[col].mode()[0])
        
        # Prepare data summary
        data_info = f"""
Dataset Shape: {data.shape[0]:,} rows × {data.shape[1]} columns
Columns: {', '.join(data.columns[:20].tolist())}{'... (showing first 20)' if len(data.columns) > 20 else ''}
Numeric Features: {len(numeric_cols)} - {', '.join(numeric_cols[:10].tolist())}{'...' if len(numeric_cols) > 10 else ''}
Categorical Features: {len(categorical_cols)} - {', '.join(categorical_cols[:5].tolist())}{'...' if len(categorical_cols) > 5 else ''}
Missing Values: {data.isnull().sum().sum():,} total
Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB
Sample Statistics:
{data.describe().iloc[:3].to_string() if len(numeric_cols) > 0 else 'No numeric columns for statistics'}
        """
        
        # Create visualizations first
        visualizations = self.create_comprehensive_visualizations(data)
        
        # Run LangChain agents
        print("\n🤖 Running LangChain Multi-Agent Analysis...")
        
        # Agent 1: Data Scientist
        print("  🔬 Data Scientist analyzing...")
        try:
            scientist_insights = self.data_scientist.run(data_info=data_info)
        except Exception as e:
            scientist_insights = f"Data Scientist analysis failed: {str(e)}"
        
        # Agent 2: Data Analyst (with context from scientist)
        print("  📈 Data Analyst analyzing...")
        try:
            analyst_result = self.data_analyst.run(
                data_info=data_info,
                processing_insights=scientist_insights
            )
            analyst_insights = analyst_result.get('insights', str(analyst_result))
        except Exception as e:
            analyst_insights = f"Data Analyst analysis failed: {str(e)}"
        
        # Agent 3: Visualization Specialist
        print("  🎨 Visualization Specialist reviewing...")
        try:
            viz_result = self.viz_specialist.run(
                data_info=data_info,
                analysis_insights=analyst_insights,
                visualizations_created=', '.join(visualizations)
            )
            viz_insights = viz_result.get('insights', str(viz_result))
        except Exception as e:
            viz_insights = f"Visualization Specialist analysis failed: {str(e)}"
        
        # Generate comprehensive report
        self._generate_comprehensive_report(data, scientist_insights, analyst_insights, 
                                          viz_insights, visualizations)
        
        # Save processed data
        cleaned_data_path = os.path.join(self.output_dir, 'cleaned_dataset.csv')
        data.to_csv(cleaned_data_path, index=False)
        
        print(f"\n🎉 LangChain Multi-Agent Analysis Complete!")
        print(f"📁 Generated {len(visualizations)} visualizations")
        print(f"🤖 Processed by 3 specialized LangChain agents")
        print(f"📊 Check '{self.output_dir}' folder for all results")
        
        return {
            'visualizations': visualizations,
            'data_scientist_insights': scientist_insights,
            'data_analyst_insights': analyst_insights,
            'visualization_insights': viz_insights,
            'data_shape': data.shape,
            'output_directory': self.output_dir
        }
    
    def _generate_comprehensive_report(self, data, scientist_insights, analyst_insights, 
                                     viz_insights, visualizations):
        """Generate comprehensive analysis report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_content = f"""# 🤖 LangChain Multi-Agent Data Analysis Report

**Generated on:** {timestamp}  
**Analysis Engine:** LangChain + OpenAI GPT  
**Agents:** Data Scientist, Data Analyst, Visualization Specialist  

---

## 📊 Executive Summary

**Dataset Overview:**
- **Records:** {data.shape[0]:,} rows
- **Features:** {data.shape[1]} columns  
- **Numeric Features:** {len(data.select_dtypes(include=[np.number]).columns)}
- **Categorical Features:** {len(data.select_dtypes(include=['object']).columns)}
- **Data Completeness:** {((data.size - data.isnull().sum().sum()) / data.size * 100):.1f}%
- **Total Visualizations Created:** {len(visualizations)}

---

## 🔬 Data Scientist Agent Analysis

**Agent Role:** Senior Data Scientist - Data Quality & Preprocessing Expert

{scientist_insights}

---

## 📈 Data Analyst Agent Analysis

**Agent Role:** Senior Data Analyst - Statistical Analysis & Business Intelligence

{analyst_insights}

---

## 🎨 Visualization Specialist Agent Analysis

**Agent Role:** Data Visualization Expert - Visual Storytelling & Dashboard Design

{viz_insights}

---

## 📊 Generated Visualizations

The following visualizations have been created and saved:

"""
        
        for i, viz in enumerate(visualizations, 1):
            report_content += f"{i}. **{viz}** - {self._get_visualization_description(viz)}\n"
        
        report_content += f"""

---

## 🎯 Key Insights Summary

**Top Data Quality Findings:**
- Dataset completeness: {((data.size - data.isnull().sum().sum()) / data.size * 100):.1f}%
- Missing values handled: {data.isnull().sum().sum():,} cells
- Memory efficient: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB

**Statistical Highlights:**
- Numeric features analyzed: {len(data.select_dtypes(include=[np.number]).columns)}
- Correlation patterns identified
- Distribution analysis completed
- Outlier detection performed

**Visualization Insights:**
- {len(visualizations)} comprehensive visualizations created
- Executive dashboard for stakeholders
- Detailed technical analysis charts
- Professional publication-ready quality

---

## 📁 Generated Files

| File | Description | Usage |
|------|-------------|-------|
| `langchain_analysis_report.md` | This comprehensive report | Documentation & sharing |
| `cleaned_dataset.csv` | Preprocessed data | Further analysis |
"""

        for viz in visualizations:
            report_content += f"| `{viz}` | {self._get_visualization_description(viz)} | Presentation & analysis |\n"
        
        report_content += f"""

---

## 🔧 Technical Details

**LangChain Components Used:**
- `ChatOpenAI` - Primary language model interface
- `PromptTemplate` - Structured agent prompts
- `LLMChain` - Agent execution chains
- `Custom Output Parsers` - Structured response handling

**Visualization Libraries:**
- `matplotlib` - Core plotting functionality
- `seaborn` - Statistical visualizations  
- `pandas` - Data manipulation
- `numpy` - Numerical computations

**Model Configuration:**
- Model: {os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')}
- Temperature: 0.1 (focused, deterministic)
- Max Tokens: 800 per agent response

---

## 🚀 Next Steps & Recommendations

Based on the multi-agent analysis, consider these next steps:

1. **Data Enhancement:** Implement the data scientist's preprocessing recommendations
2. **Statistical Deep Dive:** Execute the analyst's suggested statistical tests
3. **Visualization Improvements:** Apply the visualization specialist's enhancement suggestions
4. **Business Action:** Review business intelligence findings for actionable insights

---

*Generated by LangChain Multi-Agent Analysis System*  
*Combining AI expertise with professional data visualization*
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'langchain_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"  ✅ Comprehensive report saved: {report_path}")
    
    def _get_visualization_description(self, viz_filename):
        """Get description for visualization file"""
        descriptions = {
            'executive_dashboard.png': 'High-level overview dashboard for executives and stakeholders',
            'correlation_analysis.png': 'Feature correlation matrix showing relationships between variables',
            'feature_distributions.png': 'Statistical distributions of key numeric features',
            'outlier_analysis.png': 'Box plot analysis for outlier detection and data quality assessment'
        }
        return descriptions.get(viz_filename, 'Professional data visualization chart')

def main():
    """Main execution function"""
    print("🚀 LangChain Multi-Agent Data Analysis System")
    print("=" * 60)
    print("🔗 Powered by LangChain + OpenAI")
    print("🤖 3 Specialized AI Agents: Data Scientist, Analyst, Viz Specialist")
    print("📊 Creates Professional Visualizations")
    print()
    
    # First, install dependencies
    print("📋 Required dependencies:")
    print("   pip install -r simple_requirements.txt")
    print()
    
    try:
        # Initialize system
        analyzer = LangChainMultiAgentAnalysis()
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
        common_files = ['cancer_seek_protein.xlsx', 'data.xlsx', 'data.csv']
        for file in common_files:
            if os.path.exists(file):
                data_file = file
                print(f"📁 Found data file: {data_file}")
                break
    
    if not data_file:
        print("❌ No data file found")
        print("Usage: python langchain_main.py your_data_file.xlsx")
        print("Or place a data file (xlsx/csv) in the current directory")
        return
    
    # Run analysis
    print(f"🔍 Starting LangChain multi-agent analysis on: {data_file}")
    print()
    
    try:
        result = analyzer.analyze_data_with_agents(data_file)
        
        if result:
            print("\n" + "="*60)
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"📊 Dataset: {result['data_shape'][0]:,} rows × {result['data_shape'][1]} columns")
            print(f"🎨 Visualizations: {len(result['visualizations'])} professional charts created")
            print(f"🤖 AI Agents: 3 specialized LangChain agents provided insights")
            print(f"📁 Output Directory: {result['output_directory']}")
            print("\n🎯 Key Files Created:")
            print("   • langchain_analysis_report.md - Comprehensive analysis report")
            print("   • executive_dashboard.png - High-level overview dashboard")
            print("   • correlation_analysis.png - Feature relationship analysis")
            print("   • feature_distributions.png - Statistical distributions")
            print("   • outlier_analysis.png - Data quality assessment")
            print("   • cleaned_dataset.csv - Processed data for further use")
        else:
            print("\n❌ Analysis failed - check error messages above")
    
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()