import sqlite3
import pandas as pd
import random
from typing import Dict
import os

def load_and_shuffle_columns(sqlite_file: str) -> Dict[str, pd.DataFrame]:
    conn = sqlite3.connect(sqlite_file)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(query, conn)['name'].tolist()
    modified_tables = {}

    for table in tables:
        df = pd.read_sql(f"SELECT * FROM '{table}'", conn)

        if df.empty:
            modified_tables[table] = df
            continue

        # Shuffle the columns
        shuffled_columns = random.sample(df.columns.tolist(), len(df.columns))
        df = df[shuffled_columns]

        modified_tables[table] = df

    conn.close()
    return modified_tables


def process_databases(full_path, databases):
    # Create a dictionary to store the modified tables
    modified_tables = {}
    for db in databases:
        db_path = os.path.join(full_path, db)
        all_samples = os.listdir(db_path)
        modified_tables[db] = {}

        for sample in all_samples:
            sample_path = os.path.join(db_path, sample)
            if not os.path.isfile(sample_path):
                continue

            # Load the database and shuffle columns
            modified_tables[db][sample] = load_and_shuffle_columns(sample_path)
    return modified_tables


def get_questions_and_answers(questions_path, data_path, scale, qtype):
    questions_database = sqlite3.connect(questions_path)
    all_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", questions_database)['name'].tolist()
    questions = {}
    for table in all_tables:
        query = f"SELECT * FROM '{table}' WHERE qtype = '{qtype}' AND scale = '{scale}'"
        questions_df = pd.read_sql(query, questions_database)
        questions[table] = questions_df

    questions_database.close()
    return questions


def generate_prompts(question: str, tables: dict):
    prompt = f"Answer the question based on these tables:\n"
    for table_name, table_df in tables.items():
        prompt += f"\nTable: {table_name}\n"
        prompt += table_df.to_string(index=False)
        prompt += "\n"
    prompt += f"\nQuestion: {question}\n"
    prompt = prompt + '''This question has only one correct answer. Please
                    break down the question, evaluate each option,
                    and explain why it is correct or incorrect.
                    Conclude with your final choice on a new line
                    formatted as {answer: output}'''

    return prompt