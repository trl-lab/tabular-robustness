import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

# Import question/table processing functions from package
from src.base.base import get_questions_and_answers as get_qna_base
from src.base.base import process_databases as process_db_base

from src.missing.missing import get_questions_and_answers as get_qna_missing
from src.missing.missing import process_databases as process_db_missing

from src.shuffle.shuffle import get_questions_and_answers as get_qna_shuffle
from src.shuffle.shuffle import process_databases as process_db_shuffle


def get_package_data_dir() -> Path:
    return Path(__file__).parent / "data"


AVAILABLE_SCALES = ['1k', '2k', '4k', '6k', '8k', '16k', '32k', '64k', '128k']
DEFAULT_EXPORT_SCALES: Sequence[str] = ('1k', '2k', '4k', '6k', '8k')
PACKAGE_DATA_DIR = get_package_data_dir()
DATA_PATH = str(PACKAGE_DATA_DIR / "scaledDB")
DATABASE = str(PACKAGE_DATA_DIR / "dataset.sqlite")

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


def _tables_to_json_string(tables: Dict[str, pd.DataFrame]) -> str:
    """Convert mapping of table_name -> DataFrame into a JSON string where each
    table is converted to CSV text. Returning a string keeps the final JSONL simple.
    """
    out = {}
    for tname, df in tables.items():
        try:
            csv_text = df.to_csv(index=False)
        except Exception:
            # fallback: convert to dict
            csv_text = json.dumps(df.to_dict(orient='records'))
        out[tname] = csv_text
    return json.dumps(out)


def generate_jsonl(
    data_path: str = DATA_PATH,
    output_path: str = "questions.jsonl",
    scales: Sequence[str] | None = None,
) -> None:
    """Generate a single JSONL containing entries for all benchmark types.

    Each line will include:
      - `perturbation`: one of 'unperturbed', 'missing', or 'shuffled'
      - `scale`: the scale folder the sample originated from
    """
    answers = ['A', 'B', 'C', 'D']
    export_scales = list(scales or DEFAULT_EXPORT_SCALES)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        idx = 1
        for scale in export_scales:
            full_path = os.path.join(data_path, scale)
            if not os.path.isdir(full_path):
                print(f"Warning: scale directory not found: {full_path}. Skipping.")
                continue

            databases = sorted(entry for entry in os.listdir(full_path) if not entry.startswith('.'))
            if not databases:
                print(f"Warning: no databases discovered for scale '{scale}' under {full_path}. Skipping.")
                continue

            # --- BASE / unperturbed ---
            base_cfg = BENCHMARK_CONFIGS['base']
            base_qtypes: List[str] = list(base_cfg.get('qtypes', []))
            base_db = base_cfg.get('database', DATABASE)
            modified_tables = process_db_base(full_path, databases)

            for qtype in base_qtypes:
                questions_and_answers = get_qna_base(base_db, base_db, scale=scale, qtype=qtype)
                for db_name, question_df in questions_and_answers.items():
                    if question_df.empty:
                        continue
                    for _, row in question_df.iterrows():
                        question_text = row['question']
                        dbIdx = row['dbIdx']
                        rightIdx = int(row['rightIdx'])
                        correct_value = row[answers[rightIdx]]
                        current_row_qtype = row.get('qtype', qtype)

                        tables = modified_tables[db_name][f"{dbIdx}.sqlite"]
                        tables_str = _tables_to_json_string(tables)

                        entry = {
                            "id": idx,
                            "scale": scale,
                            "perturbation": "unperturbed",
                            "qtype": current_row_qtype,
                            "question": str(question_text),
                            "tables": tables_str,
                            "correct answer": str(correct_value)
                        }
                        out_f.write(json.dumps(entry) + "\n")
                        idx += 1

            # --- MISSING ---
            missing_cfg = BENCHMARK_CONFIGS['missing']
            missing_qtypes: List[str] = list(missing_cfg.get('qtypes', []))
            missing_db = missing_cfg.get('database', DATABASE)
            modified_tables = process_db_missing(full_path, databases)

            questions_and_answers_by_qtype = get_qna_missing(
                missing_db, missing_db, scale=scale, qtypes=missing_qtypes
            )
            for current_qtype in missing_qtypes:
                db_map = questions_and_answers_by_qtype.get(current_qtype, {})
                for db_name, question_df in db_map.items():
                    if question_df.empty:
                        continue
                    for _, row in question_df.iterrows():
                        question_text = row['question']
                        dbIdx = row['dbIdx']
                        rightIdx = int(row['rightIdx'])
                        correct_value = row[answers[rightIdx]]

                        tables = modified_tables[db_name][f"{dbIdx}.sqlite"]
                        tables_str = _tables_to_json_string(tables)

                        entry = {
                            "id": idx,
                            "scale": scale,
                            "perturbation": "missing",
                            "qtype": current_qtype,
                            "question": str(question_text),
                            "tables": tables_str,
                            "correct answer": str(correct_value)
                        }
                        out_f.write(json.dumps(entry) + "\n")
                        idx += 1

            # --- SHUFFLE ---
            shuffle_cfg = BENCHMARK_CONFIGS['shuffle']
            shuffle_qtypes: List[str] = list(shuffle_cfg.get('qtypes', []))
            shuffle_db = shuffle_cfg.get('database', DATABASE)
            modified_tables = process_db_shuffle(full_path, databases)

            for qtype in shuffle_qtypes:
                questions_and_answers = get_qna_shuffle(shuffle_db, shuffle_db, scale=scale, qtype=qtype)
                for db_name, question_df in questions_and_answers.items():
                    if question_df.empty:
                        continue
                    for _, row in question_df.iterrows():
                        question_text = row['question']
                        dbIdx = row['dbIdx']
                        rightIdx = int(row['rightIdx'])
                        correct_value = row[answers[rightIdx]]
                        current_row_qtype = row.get('qtype', qtype)

                        tables = modified_tables[db_name][f"{dbIdx}.sqlite"]
                        tables_str = _tables_to_json_string(tables)

                        entry = {
                            "id": idx,
                            "scale": scale,
                            "perturbation": "shuffled",
                            "qtype": current_row_qtype,
                            "question": str(question_text),
                            "tables": tables_str,
                            "correct answer": str(correct_value)
                        }
                        out_f.write(json.dumps(entry) + "\n")
                        idx += 1


def _parse_args():
    p = argparse.ArgumentParser(description="Export benchmark questions to JSONL")
    p.add_argument('--data-path', default=DATA_PATH, help='Path to scaledDB root')
    p.add_argument('--out', '-o', default='questions.jsonl', help='Output JSONL path')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    # Always generate all perturbation variants across the default scales into a single JSONL
    joined_scales = ", ".join(DEFAULT_EXPORT_SCALES)
    print(f"Generating JSONL for all perturbations across scales [{joined_scales}] -> {args.out}")
    generate_jsonl(data_path=args.data_path, output_path=args.out)
