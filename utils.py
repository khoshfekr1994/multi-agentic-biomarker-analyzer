"""
Utility functions for the multi-agent data analysis system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

class DataUtils:
    """Utility functions for data operations"""
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load data from various file formats"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension == '.xlsx' or file_extension == '.xls':
                return pd.read_excel(file_path)
            elif file_extension == '.json':
                return pd.read_json(file_path)
            elif file_extension == '.parquet':
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    @staticmethod
    def basic_info(df: pd.DataFrame) -> dict:
        """Get basic information about the dataframe"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        return info
    
    @staticmethod
    def clean_data(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """Basic data cleaning operations"""
        df_clean = df.copy()
        
        if strategy == 'auto':
            # Remove completely empty rows and columns
            df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
            
            # Handle missing values based on data type
            for col in df_clean.columns:
                if df_clean[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    # Fill categorical columns with mode
                    mode_value = df_clean[col].mode()
                    if not mode_value.empty:
                        df_clean[col].fillna(mode_value[0], inplace=True)
        
        return df_clean
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: list = None) -> dict:
        """Detect outliers using IQR method"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = {
                'count': outlier_mask.sum(),
                'percentage': (outlier_mask.sum() / len(df)) * 100,
                'indices': df[outlier_mask].index.tolist()
            }
        
        return outliers

class VisualizationUtils:
    """Utility functions for creating visualizations"""
    
    @staticmethod
    def setup_style():
        """Setup consistent styling for matplotlib plots"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    @staticmethod
    def save_plot(fig, filename: str, output_dir: str = "outputs"):
        """Save plot to file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")
    
    @staticmethod
    def create_summary_dashboard(df: pd.DataFrame, output_dir: str = "outputs"):
        """Create a summary dashboard with multiple plots"""
        VisualizationUtils.setup_style()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Summary Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Missing values heatmap
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_data = missing_data[missing_data > 0]
            axes[0, 0].bar(range(len(missing_data)), missing_data.values)
            axes[0, 0].set_xticks(range(len(missing_data)))
            axes[0, 0].set_xticklabels(missing_data.index, rotation=45)
            axes[0, 0].set_title('Missing Values by Column')
            axes[0, 0].set_ylabel('Count')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
            axes[0, 0].set_title('Missing Values by Column')
        
        # Plot 2: Data types distribution
        dtype_counts = df.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Data Types Distribution')
        
        # Plot 3: Numeric columns correlation (if any)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
            axes[1, 0].set_title('Correlation Matrix')
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient Numeric Columns', ha='center', va='center')
            axes[1, 0].set_title('Correlation Matrix')
        
        # Plot 4: Dataset shape info
        info_text = f"""
        Dataset Shape: {df.shape}
        Total Rows: {df.shape[0]:,}
        Total Columns: {df.shape[1]}
        Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        Duplicate Rows: {df.duplicated().sum()}
        """
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Dataset Information')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'summary_dashboard.png', output_dir)
        return fig

class ReportUtils:
    """Utility functions for generating reports"""
    
    @staticmethod
    def generate_markdown_report(results: dict, output_dir: str = "outputs"):
        """Generate a comprehensive markdown report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Data Analysis Report
        
Generated on: {timestamp}

## Executive Summary
{results.get('summary', 'No summary provided')}

## Data Processing Results
{results.get('processing', 'No processing results')}

## Statistical Analysis
{results.get('analysis', 'No analysis results')}

## Visualizations
{results.get('visualizations', 'No visualizations created')}

## Recommendations
{results.get('recommendations', 'No recommendations provided')}

---
*Report generated by Multi-Agent Data Analysis System*
"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        report_path = os.path.join(output_dir, 'analysis_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
        return report_path