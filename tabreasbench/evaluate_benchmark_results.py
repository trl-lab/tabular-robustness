import re
from ollama import chat
import csv
import pandas as pd
from tqdm import tqdm
import os
import argparse
from typing import Dict, List, Tuple

def extract_answer(text: str) -> str:
    """Extract the yes/no answer from the LLM's response."""
    try:
        answer = text.lower().replace('*', '').split('answer correct: ')[1]
        return answer.strip()
    except:
        return None

def evaluate_answer(question_text: str, response_text: str, correct_answer: str, benchmark_type: str) -> str:
    """Use LLM to evaluate if the response matches the correct answer."""
    # Customize the prompt based on benchmark type
    if benchmark_type == 'missing':
        context = "Consider that if the answer is None or missing, it means that the value could not be found in the table."
    elif benchmark_type == 'shuffle':
        context = "Consider that the table rows might be in a different order than expected."
    else:  # base benchmark
        context = "Consider the exact numerical or categorical value that should be returned."

    llm_judge_question = f"""When it comes to the following question: '{question_text}',
                            does this answer '{response_text}' match the value of the correct answer '{correct_answer}'?
                            
                            {context}
                            
                            Please conclude your answer with 'answer correct: yes/no'"""
    
    response_judge = chat(model='qwen2.5:32b', messages=[{'role': 'user', 'content': llm_judge_question}])
    return extract_answer(response_judge['message']['content'])

def perform_test(question_text: str, response_text: str, correct_answer: str, benchmark_type: str) -> Tuple[bool, str]:
    """Evaluates the answer using the LLM and returns the evaluation result."""
    # Clean up the question text
    question_text = re.sub(r'\s+', ' ', question_text).strip()
    
    # Evaluate the answer
    evaluation_result = evaluate_answer(question_text, response_text, correct_answer, benchmark_type)

    if evaluation_result and 'yes' in evaluation_result:
        return True, response_text
    elif evaluation_result and 'no' in evaluation_result:
        return False, response_text
    else:
        return False, "evaluation_failed"

def evaluate_benchmark_results(input_file: str, output_file: str = None) -> None:
    """
    Evaluates the benchmark results from a CSV file and saves the evaluation results.
    
    Args:
        input_file: Path to the input CSV file containing benchmark results
        output_file: Path to save the evaluation results (defaults to input_file_evaluated.csv)
    """
    if output_file is None:
        output_file = input_file.replace('.csv', '_evaluated.csv')

    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Prepare output file
    fieldnames = list(df.columns) + ['evaluation_result']
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Process each row
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating results"):
            try:
                is_correct, evaluated_answer = perform_test(
                    question_text=row['question'],
                    response_text=row['extracted_answer'],
                    correct_answer=str(row['correct_answer']),
                    benchmark_type=row['benchmark_type']
                )

                # Create output row
                output_row = row.to_dict()
                output_row['evaluation_result'] = 'yes' if is_correct else 'no'
                
                writer.writerow(output_row)

            except Exception as e:
                print(f"Error processing question: {row['question']} - {e}")
                output_row = row.to_dict()
                output_row['evaluation_result'] = 'error'
                writer.writerow(output_row) 