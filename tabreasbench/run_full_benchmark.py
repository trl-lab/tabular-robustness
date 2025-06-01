import os
import argparse
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm

from run_benchmark import run_benchmark_suite, AVAILABLE_LLMS
from evaluate_benchmark_results import evaluate_benchmark_results
from aggregate_evaluation_results import generate_combined_table, save_results, generate_latex_tables

def setup_output_structure(output_dir: str, model_name: str) -> dict:
    """
    Creates the necessary directory structure for the benchmark results.
    Returns a dictionary with all relevant paths.
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    raw_results_dir = os.path.join(output_dir, "raw_results")
    evaluated_results_dir = os.path.join(output_dir, "evaluated_results")
    aggregated_results_dir = os.path.join(output_dir, "aggregated_results")
    
    os.makedirs(raw_results_dir, exist_ok=True)
    os.makedirs(evaluated_results_dir, exist_ok=True)
    os.makedirs(aggregated_results_dir, exist_ok=True)
    
    return {
        "raw_results": raw_results_dir,
        "evaluated_results": evaluated_results_dir,
        "aggregated_results": aggregated_results_dir
    }

def run_full_benchmark(model_name: str, output_dir: str) -> None:
    """
    Executes the complete benchmark workflow:
    1. Run benchmarks
    2. Evaluate results
    3. Aggregate results
    """
    # Setup directory structure
    dirs = setup_output_structure(output_dir, model_name)
    
    # Add model to available LLMs if not present
    if model_name not in AVAILABLE_LLMS:
        AVAILABLE_LLMS[model_name] = {'use_openai': False}
    
    print(f"\n=== Starting benchmark run for model: {model_name} ===")
    
    # Step 1: Run benchmarks
    print("\n1. Running benchmarks...")
    run_benchmark_suite()
    
    # Step 2: Evaluate results
    print("\n2. Evaluating results...")
    raw_results_file = os.path.join(dirs["raw_results"], f"results_{model_name}.csv")
    evaluated_results_file = os.path.join(dirs["evaluated_results"], f"results_{model_name}_evaluated.csv")
    
    if os.path.exists(raw_results_file):
        evaluate_benchmark_results(raw_results_file, evaluated_results_file)
    else:
        print(f"Error: Raw results file not found at {raw_results_file}")
        return
    
    # Step 3: Aggregate results
    print("\n3. Aggregating results...")
    csv_files = [evaluated_results_file]
    
    if os.path.exists(evaluated_results_file):
        # Generate combined tables
        overall_summary, detailed_df = generate_combined_table(csv_files)
        
        # Generate LaTeX tables
        overall_latex, detailed_latex = generate_latex_tables(overall_summary, detailed_df)
        
        # Save results
        save_results(
            overall_summary=overall_summary,
            detailed_df=detailed_df,
            overall_latex=overall_latex,
            detailed_latex=detailed_latex,
            output_dir=dirs["aggregated_results"]
        )
        
        # Print summary
        print("\nOverall Summary Table:")
        print(overall_summary.to_string())
        print("\nDetailed Results Table:")
        print(detailed_df.to_string())
    else:
        print(f"Error: Evaluated results file not found at {evaluated_results_file}")
        return
    
    print(f"\n=== Benchmark workflow completed ===")
    print(f"Results have been saved to: {output_dir}")
    print(f"- Raw results: {dirs['raw_results']}")
    print(f"- Evaluated results: {dirs['evaluated_results']}")
    print(f"- Aggregated results: {dirs['aggregated_results']}")