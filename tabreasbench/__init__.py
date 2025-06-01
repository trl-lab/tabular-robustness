"""
TabReasBench - A benchmark suite for evaluating table reasoning capabilities of language models.
"""

from .run_benchmark import run_benchmark_suite
from .evaluate_benchmark_results import evaluate_benchmark_results
from .aggregate_evaluation_results import (
    generate_combined_table,
    generate_latex_tables,
    save_results
)

__version__ = "0.1.0"
__all__ = [
    'run_benchmark_suite',
    'evaluate_benchmark_results',
    'generate_combined_table',
    'generate_latex_tables',
    'save_results'
] 