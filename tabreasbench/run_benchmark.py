import sqlite3
import pandas as pd
import random
import os
import json
from tqdm import tqdm
import csv
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# Import functions from different benchmark types
from .src.base.base import get_questions_and_answers as get_qna_base
from .src.base.base import process_databases as process_db_base
from .src.base.base import generate_prompts as generate_prompts_base

from .src.missing.missing import get_questions_and_answers as get_qna_missing
from .src.missing.missing import process_databases as process_db_missing
from .src.missing.missing import generate_prompts as generate_prompts_missing

from .src.shuffle.shuffle import get_questions_and_answers as get_qna_shuffle
from .src.shuffle.shuffle import process_databases as process_db_shuffle
from .src.shuffle.shuffle import generate_prompts as generate_prompts_shuffle

from .src.llm_interface import get_llm_response, get_openai_response

def get_package_data_dir() -> Path:
    """Get the absolute path to the package's data directory."""
    return Path(__file__).parent / 'data'

# Define available LLMs and their configurations
AVAILABLE_LLMS = {
    'ollama': {'use_openai': False},
    'openai': {'use_openai': True}
}

# Define benchmark constants
AVAILABLE_SCALES = ['1k', '2k', '4k', '6k', '8k', '16k', '32k', '64k', '128k']
PACKAGE_DATA_DIR = get_package_data_dir()
DATABASE = str(PACKAGE_DATA_DIR / "dataset.sqlite")
DATA_PATH = str(PACKAGE_DATA_DIR / "scaledDB")

# Define benchmark types and their specific configurations
BENCHMARK_CONFIGS = {
    'base': {
        'qtypes': ['count', 'average', 'sum', 'item_select', 'row_match', 'difference'],
        'database': DATABASE
    },
    'missing': {
        'qtypes': ['average_missing', 'sum_missing'],
        'database': DATABASE
    },
    'shuffle': {
        'qtypes': ['count'],
        'database': DATABASE
    }
}

def run_benchmark(
    llm: str,
    benchmark_type: str,
    scale: str,
    qtype: str | list[str] | None = None,
    data_path: str = DATA_PATH,
    progressbar: tqdm = None,
    output_dir: str = "testResults"
) -> None:
    """
    Runs a specific benchmark type with the given parameters and writes results continuously to CSV.

    Args:
        llm: The language model to use (e.g., "qwen2.5:32b").
        benchmark_type: The type of benchmark to run ('base', 'missing', 'shuffle').
        scale: The scale of the dataset.
        qtype: The question type(s) to include (for 'missing' and 'shuffle').
        data_path: The base path to the scaled databases.
        progressbar: Optional tqdm progress bar for tracking overall progress.
        output_dir: Directory to save results in.
    """
    if benchmark_type not in BENCHMARK_CONFIGS:
        print(f"Error: Unknown benchmark type '{benchmark_type}'")
        return

    # Determine if we should use OpenAI based on the model name
    use_openai = llm.startswith('gpt-')
    
    # Setup CSV file for this run
    output_path = os.path.join(output_dir, f"results_{llm.replace(':', '_')}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_exists = os.path.isfile(output_path)
    
    # Initialize CSV writer
    csvfile = open(output_path, 'a', newline='')
    fieldnames = ["benchmark_type", "scale", "qtype", "question", "correct_answer", 
                 "extracted_answer", "correct_index", "llm"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    config = BENCHMARK_CONFIGS[benchmark_type]
    qna_database = config['database']
    
    print(f"Running {benchmark_type} benchmark for LLM: {llm}, Scale: {scale}, Qtype: {qtype}")

    questions_and_answers = None
    modified_tables = None
    answers = ['A', 'B', 'C', 'D']

    # --- Data Loading and Processing --- #
    if benchmark_type == 'base':
        questions_and_answers = get_qna_base(qna_database, qna_database, scale=scale, qtype=qtype)
        full_path = os.path.join(data_path, scale)
        databases = os.listdir(full_path)
        modified_tables = process_db_base(full_path, databases)

    elif benchmark_type == 'missing':
        questions_and_answers_by_qtype = get_qna_missing(qna_database, qna_database, scale=scale, qtypes=qtype)
        full_path = os.path.join(data_path, scale)
        databases = os.listdir(full_path)
        modified_tables = process_db_missing(full_path, databases)
        
        if isinstance(qtype, str):
            questions_and_answers = {qtype: questions_and_answers_by_qtype.get(qtype, {})}
        elif isinstance(qtype, list):
            questions_and_answers = {qt: questions_and_answers_by_qtype.get(qt, {}) for qt in qtype}
        else:
            print("Warning: qtype not specified for missing benchmark. Processing all available qtypes.")
            questions_and_answers = questions_and_answers_by_qtype

    elif benchmark_type == 'shuffle':
        questions_and_answers = get_qna_shuffle(qna_database, qna_database, scale=scale, qtype=qtype)
        full_path = os.path.join(data_path, scale)
        databases = os.listdir(full_path)
        modified_tables = process_db_shuffle(full_path, databases)

    if not questions_and_answers or not modified_tables:
        print("Failed to load data or process databases.")
        return

    # --- Benchmark Execution --- #
    total_samples = sum(len(df) for df in questions_and_answers.values())
    if benchmark_type == 'missing':
        total_samples = sum(len(df) for qtype_data in questions_and_answers.values() for df in qtype_data.values())

    if progressbar is None:
        progressbar = tqdm(total=total_samples, desc=f"{benchmark_type} Progress", unit="sample")
        progressbar.n = 0

    try:
        if benchmark_type == 'missing':
            for current_qtype, db_qna_data in questions_and_answers.items():
                for db_name, question_df in db_qna_data.items():
                    for index, row in question_df.iterrows():
                        progressbar.update(1)
                        progressbar.refresh()
                        
                        question_text = row['question']
                        dbIdx = row['dbIdx']
                        rightIdx = row['rightIdx']
                        row_index = row['row_index']
                        table_name = row['table_name']
                        column_name = row['column_name']
                        correct_value = np.float64(row[answers[rightIdx]])

                        tables = modified_tables[db_name][f"{dbIdx}.sqlite"]

                        if "Food" in table_name:
                            table_name = table_name.replace("_", "-")
                        if table_name in tables and column_name in tables[table_name].columns:
                            tables[table_name].at[row_index, column_name] = None
                        else:
                            print(f"Warning: Table '{table_name}' or column '{column_name}' not found in {db_name}/{dbIdx}.sqlite")

                        prompt = generate_prompts_missing(question_text, tables)
                        response_text = get_openai_response(prompt) if use_openai else get_llm_response(prompt, model=llm)
                        response_text = response_text.replace("\n", " ").replace(",", " ")

                        cur_result = {
                            "benchmark_type": benchmark_type,
                            "scale": scale,
                            "qtype": current_qtype,
                            "question": question_text,
                            "correct_answer": correct_value,
                            "extracted_answer": response_text,
                            "correct_index": rightIdx,
                            "llm": llm
                        }
                        writer.writerow(cur_result)
                        csvfile.flush()  # Ensure results are written immediately
        else:  # base and shuffle benchmarks
            for db_name, question_df in questions_and_answers.items():
                for index, row in question_df.iterrows():
                    progressbar.update(1)
                    progressbar.refresh()
                    
                    question_text = row['question']
                    dbIdx = row['dbIdx']
                    rightIdx = row['rightIdx']
                    correct_value = row[answers[rightIdx]]
                    current_qtype = row.get('qtype', qtype)

                    tables = modified_tables[db_name][f"{dbIdx}.sqlite"]

                    if benchmark_type == 'base':
                        prompt = generate_prompts_base(question_text, tables)
                    elif benchmark_type == 'shuffle':
                        prompt = generate_prompts_shuffle(question_text, tables)

                    response_text = get_openai_response(prompt) if use_openai else get_llm_response(prompt, model=llm)
                    response_text = response_text.replace("\n", " ").replace(",", " ")

                    cur_result = {
                        "benchmark_type": benchmark_type,
                        "scale": scale,
                        "qtype": current_qtype,
                        "question": question_text,
                        "correct_answer": correct_value,
                        "extracted_answer": response_text,
                        "correct_index": rightIdx,
                        "llm": llm
                    }
                    writer.writerow(cur_result)
                    csvfile.flush()  # Ensure results are written immediately
    finally:
        csvfile.close()

def run_benchmark_suite(model: str, output_dir: str = "benchmark_results") -> None:
    """
    Runs the complete benchmark suite across all benchmark types, scales, and the specified model.
    This function orchestrates the execution of all benchmarks and manages the overall progress tracking.
    
    Args:
        model: The name of the model to use with Ollama
        output_dir: Directory to save results in
    """
    # Calculate total number of samples across all benchmarks
    total_samples = 0
    for benchmark_type, config in BENCHMARK_CONFIGS.items():
        qna_database = config['database']
        for scale in AVAILABLE_SCALES:
            if benchmark_type in ['missing', 'shuffle']:
                for qtype in config['qtypes']:
                    # Load questions to count samples
                    if benchmark_type == 'missing':
                        questions = get_qna_missing(qna_database, qna_database, scale=scale, qtypes=[qtype])
                        total_samples += sum(len(df) for qtype_data in questions.values() for df in qtype_data.values())
                    else:  # shuffle
                        questions = get_qna_shuffle(qna_database, qna_database, scale=scale, qtype=qtype)
                        total_samples += sum(len(df) for df in questions.values())
            else:  # base benchmark
                for qtype in config['qtypes']:
                    questions = get_qna_base(qna_database, qna_database, scale=scale, qtype=qtype)
                    total_samples += sum(len(df) for df in questions.values())

    # Create a single progress bar for the entire benchmark suite
    with tqdm(total=total_samples, desc="Overall Benchmark Progress", unit="sample") as progressbar:
        # Run each benchmark type for each scale
        for benchmark_type, config in BENCHMARK_CONFIGS.items():
            for scale in AVAILABLE_SCALES:
                if benchmark_type in ['missing', 'shuffle']:
                    for qtype in config['qtypes']:
                        qtype_param = qtype if benchmark_type == 'shuffle' else [qtype]
                        run_benchmark(
                            llm=model,
                            benchmark_type=benchmark_type,
                            scale=scale,
                            qtype=qtype_param,
                            output_dir=output_dir,
                            progressbar=progressbar
                        )
                else:  # base benchmark
                    for qtype in config['qtypes']:
                        run_benchmark(
                            llm=model,
                            benchmark_type=benchmark_type,
                            scale=scale,
                            qtype=qtype,
                            output_dir=output_dir,
                            progressbar=progressbar
                        )