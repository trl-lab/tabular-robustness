import argparse
import os
from pathlib import Path
from typing import Optional

from .run_benchmark import run_benchmark_suite
from .evaluate_benchmark_results import evaluate_benchmark_results
from .aggregate_evaluation_results import generate_combined_table, generate_latex_tables, save_results

def main():
    """Main entry point for the tabreasbench command."""
    parser = argparse.ArgumentParser(description='Run table reasoning benchmarks')
    parser.add_argument('--model', required=True, help='Name of the model to use with Ollama (e.g., "qwen2.5:32b")')
    parser.add_argument('--output_dir', default='benchmark_results', help='Directory to save all results')
    
    args = parser.parse_args()
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    raw_results_dir = output_dir / 'raw_results'
    evaluated_results_dir = output_dir / 'evaluated_results'
    aggregated_results_dir = output_dir / 'aggregated_results'
    
    for dir_path in [raw_results_dir, evaluated_results_dir, aggregated_results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Run the benchmark suite
    print(f"Running benchmarks with model: {args.model}")
    print(f"Results will be saved to: {output_dir}")
    
    # Run benchmarks and save raw results
    run_benchmark_suite(model=args.model, output_dir=str(raw_results_dir))
    
    # Evaluate results
    print("\nEvaluating benchmark results...")
    raw_results_file = raw_results_dir / f"results_{args.model}.csv"
    evaluated_results_file = evaluated_results_dir / f"results_{args.model}_evaluated.csv"
    
    if raw_results_file.exists():
        evaluate_benchmark_results(str(raw_results_file), str(evaluated_results_file))
        
        # Aggregate results
        print("\nAggregating evaluation results...")
        overall_summary, detailed_df = generate_combined_table([str(evaluated_results_file)])
        overall_latex, detailed_latex = generate_latex_tables(overall_summary, detailed_df)
        
        # Save aggregated results
        save_results(
            overall_summary=overall_summary,
            detailed_df=detailed_df,
            overall_latex=overall_latex,
            detailed_latex=detailed_latex,
            output_dir=str(aggregated_results_dir)
        )
        
        print("\nBenchmark suite completed successfully!")
        print(f"Results are available in: {output_dir}")
    else:
        print(f"Error: No raw results found at {raw_results_file}")

if __name__ == '__main__':
    main() 