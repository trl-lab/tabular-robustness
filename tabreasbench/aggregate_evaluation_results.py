import pandas as pd
from typing import Dict, List, Tuple
import os
import argparse
from pathlib import Path

def calculate_accuracy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate accuracy metrics for the evaluation results.
    Filters out 'error' results and calculates accuracy percentages.
    """
    # Filter out evaluation errors
    df_filtered = df[df['evaluation_result'].isin(['yes', 'no'])]
    
    # Calculate metrics by grouping
    summary = df_filtered.groupby(['benchmark_type', 'scale', 'qtype']).agg(
        total=('evaluation_result', 'size'),
        correct=('evaluation_result', lambda x: (x == 'yes').sum()),
        errors=('evaluation_result', lambda x: (x == 'error').sum())
    ).reset_index()
    
    # Calculate accuracy percentage
    summary['accuracy'] = (summary['correct'] / summary['total']) * 100
    
    return summary

def generate_combined_table(csv_files: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate two combined tables:
    1. Overall accuracy by benchmark type and scale
    2. Detailed accuracy by benchmark type, scale, and question type
    """
    all_data = []
    
    for file_path in csv_files:
        # Extract model name from filename (assuming format: results_MODEL_evaluated.csv)
        model_name = Path(file_path).stem.split('_')[1]
        
        # Read and process each CSV file
        df = pd.read_csv(file_path)
        df['model'] = model_name
        
        # Calculate metrics
        metrics_df = calculate_accuracy_metrics(df)
        metrics_df['model'] = model_name
        
        all_data.append(metrics_df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create overall summary (by model, benchmark type, and scale)
    overall_summary = combined_df.groupby(['model', 'benchmark_type', 'scale']).agg(
        total_samples=('total', 'sum'),
        total_correct=('correct', 'sum'),
        total_errors=('errors', 'sum')
    ).reset_index()
    
    overall_summary['accuracy'] = (overall_summary['total_correct'] / overall_summary['total_samples']) * 100
    
    return overall_summary, combined_df

def generate_latex_tables(overall_summary: pd.DataFrame, detailed_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Generate LaTeX tables for both overall summary and detailed results.
    """
    # Overall summary table
    overall_latex = overall_summary.pivot_table(
        index=['model', 'benchmark_type'],
        columns='scale',
        values='accuracy',
        aggfunc='mean'
    ).round(2).to_latex()
    
    # Detailed table (by qtype)
    detailed_latex = detailed_df.pivot_table(
        index=['model', 'benchmark_type', 'qtype'],
        columns='scale',
        values='accuracy',
        aggfunc='mean'
    ).round(2).to_latex()
    
    return overall_latex, detailed_latex

def save_results(overall_summary: pd.DataFrame, detailed_df: pd.DataFrame, 
                overall_latex: str, detailed_latex: str, output_dir: str) -> None:
    """Save all results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV files
    overall_summary.to_csv(os.path.join(output_dir, 'overall_summary.csv'), index=False)
    detailed_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    # Save LaTeX tables
    with open(os.path.join(output_dir, 'overall_summary.tex'), 'w') as f:
        f.write(overall_latex)
    with open(os.path.join(output_dir, 'detailed_results.tex'), 'w') as f:
        f.write(detailed_latex) 