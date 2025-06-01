import sqlite3
import pandas as pd
import os
from typing import Dict


def load_tables(sqlite_file: str) -> Dict[str, pd.DataFrame]:
    """Load all tables from a SQLite database without modifying rows."""
    conn = sqlite3.connect(sqlite_file)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(query, conn)['name'].tolist()
    loaded_tables = {table: pd.read_sql(f"SELECT * FROM '{table}'", conn) for table in tables}
    conn.close()
    return loaded_tables


def process_databases(full_path: str, databases: list) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """Process all SQLite databases in given directories and load their tables."""
    processed_data = {}
    for db in databases:
        if ".DS_Store" in db:
            continue
        db_path = os.path.join(full_path, db)
        all_samples = os.listdir(db_path)
        processed_data[db] = {}

        for sample in all_samples:
            sample_path = os.path.join(db_path, sample)
            if not os.path.isfile(sample_path):
                continue

            processed_data[db][sample] = load_tables(sample_path)
    return processed_data


def get_questions_and_answers(questions_path: str, data_path: str, scale: str, qtype: str) -> Dict[str, pd.DataFrame]:
    """Fetch filtered questions from the questions SQLite database."""
    conn = sqlite3.connect(questions_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(query, conn)['name'].tolist()

    questions = {}
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM '{table}' WHERE scale = ? AND qtype = ?", conn, params=(scale, qtype))
        questions[table] = df

    conn.close()
    return questions


def generate_prompts(question: str, tables: Dict[str, pd.DataFrame]) -> str:
    """Generate a detailed prompt string using provided tables and a question."""
    prompt = "Answer the question based on these tables:\n"
    for table_name, table_df in tables.items():
        prompt += f"\nTable: {table_name}\n{table_df.to_csv()}\n"
    
    prompt += f"\nQuestion: {question}\n"
    prompt += """This question has only one correct answer. Please
                break down the question, evaluate each option,
                and explain why it is correct or incorrect.
                Conclude with your final answer."""
    return prompt
