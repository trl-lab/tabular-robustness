import argparse
import json
import math
import random
import re
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
from tqdm import tqdm

try:
    # Optional import for loading datasets from Hugging Face
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None

from src.llm_interface import get_llm_response, get_openai_response


def should_use_openai(model_name: str) -> bool:
    """Return True when the provided model should be queried via OpenAI."""
    return model_name.startswith("gpt-") and "oss" not in model_name


# Local JSONL loading removed. Dataset is loaded from Hugging Face via
# `load_samples_from_hf` only.


def load_samples_from_hf(dataset_id: str, split: str = "test", hf_token: str | None = None) -> Iterable[Dict[str, Any]]:
    """Yield samples from a Hugging Face dataset.

    The function expects the dataset to contain fields compatible with the
    benchmark: `id`, `question`, `tables`, `correct_answer`, `qtype`, `scale`,
    and `benchmark_type`. It yields dicts with the same keys as `load_samples`.
    If the `datasets` library is not installed an informative RuntimeError is raised.
    """
    if load_dataset is None:
        raise RuntimeError(
            "The 'datasets' library is required to load Hugging Face datasets. "
            "Install it with: pip install datasets"
        )

    # Pass token to `load_dataset` if provided to access private datasets.
    load_kwargs = {}

    try:
        ds = load_dataset(dataset_id, split=split, token=hf_token)
    except ValueError as exc:
        raise RuntimeError(
            f"Failed to load Hugging Face dataset '{dataset_id}' (split='{split}'). "
            "If the dataset is private, ensure --hf-token is set and valid, and verify the split name."
        ) from exc

    for item in ds:
        # Convert to plain dict and provide compatibility with existing code
        record = dict(item)
        # Some datasets use different field names for the correct answer
        if "correct answer" in record and "correct_answer" not in record:
            record["correct_answer"] = record["correct answer"]
        yield record


def sample_records_by_share(
    records: Iterable[Dict[str, Any]], share: float, rng: random.Random | None = None
) -> list[Dict[str, Any]]:
    """Return a random subset of records keeping up to `share` of each (scale, qtype)."""

    if share <= 0.0:
        return []
    records_list = list(records)
    if share >= 1.0:
        return records_list

    rng = rng or random.Random()
    groups: Dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for record in records_list:
        key = (
            record.get("scale", "unknown"),
            record.get("qtype", "unknown"),
        )
        groups[key].append(record)

    sampled: list[Dict[str, Any]] = []
    for group_records in groups.values():
        total = len(group_records)
        quota = min(total, math.ceil(total * share))
        if quota <= 0:
            continue
        if quota >= total:
            sampled.extend(group_records)
        else:
            sampled.extend(rng.sample(group_records, quota))

    rng.shuffle(sampled)
    return sampled


def format_tables(tables_blob: str) -> str:
    """Turn the JSON-encoded table payload into a readable text block."""
    prompt_sections = ["Answer the question based on these tables:"]
    try:
        tables = json.loads(tables_blob)
    except json.JSONDecodeError:
        prompt_sections.append(tables_blob)
        return "\n\n".join(prompt_sections)

    for table_name, csv_payload in tables.items():
        table_block = csv_payload
        try:
            dataframe = pd.read_csv(StringIO(csv_payload))
            table_block = dataframe.to_csv(index=False)
        except Exception:
            # Fall back to the raw CSV payload when pandas cannot parse it.
            table_block = csv_payload
        prompt_sections.append(f"Table: {table_name}\n{table_block}")
    return "\n\n".join(prompt_sections)


def build_question_prompt(question: str, tables_blob: str) -> str:
    table_text = format_tables(tables_blob)
    instructions = (
        "This question has only one correct answer. "
        "Work step-by-step, explain the reasoning briefly, "
        "and finish with the final answer on the last line. "
        "Ignore any outside or internal knowledge; base the answer solely on the information in the tables."
    )
    return f"{table_text}\n\nQuestion: {question}\n{instructions}"


def build_judge_prompt(question: str, answer: str, correct: str, benchmark_type: str) -> str:
    context_lookup = {
        "missing": "Consider that None indicates a missing value in the table.",
        "shuffle": "Rows might appear in a shuffled order across tables.",
    }
    context = context_lookup.get(benchmark_type, "Compare the factual correctness of the answer.")
    return (
        f"When it comes to the following question: '{question}', does this answer "
        f"'{answer}' match the value of the correct answer '{correct}'?\n\n"
        f"{context}\n"
        "Use only the provided question, answer, and ground-truth value; "
        "ignore any internal or external knowledge even if it conflicts.\n\n"
        "Please conclude your answer with 'answer correct: yes/no'"
    )


def call_model(prompt: str, model: str, use_openai: bool) -> str:
    if use_openai:
        return get_openai_response(prompt, model=model)
    return get_llm_response(prompt, model=model)


def extract_judge_decision(text: str) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"[\*\s]+", " ", text.lower())
    match = re.search(r"answer correct:\s*(yes|no)", cleaned)
    return match.group(1) if match else None


def aggregate_counts(counter: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for key, values in counter.items():
        total = values.get("total", 0)
        correct = values.get("correct", 0)
        summary[key] = {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total * 100) if total else 0.0,
        }
    return summary


def parse_share(value: str) -> float:
    share = float(value)
    if not 0.0 <= share <= 1.0:
        raise argparse.ArgumentTypeError("--share must be between 0 and 1")
    return share


def run_simple_benchmark(
    output_path: Path,
    summary_path: Path,
    model: str,
    judge_model: str,
    force_openai: bool,
    force_judge_openai: bool,
    max_samples: int | None,
    hf_dataset: str,
    hf_split: str = "train",
    hf_token: str | None = None,
    share: float = 1.0,
) -> None:
    # Always load from Hugging Face dataset (the CLI controls which dataset).
    samples = list(load_samples_from_hf(hf_dataset, split=hf_split, hf_token=hf_token))
    samples = sample_records_by_share(samples, share)
    if max_samples is not None:
        samples = samples[:max_samples]

    use_openai = force_openai or should_use_openai(model)
    judge_use_openai = force_judge_openai or should_use_openai(judge_model)

    overall = {"total": 0, "correct": 0, "incorrect": 0, "evaluation_failed": 0}
    by_qtype: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_scale: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    # Phase 1: Generate answers
    for sample in tqdm(samples, desc="Answering questions"):
        benchmark_type = sample.get("benchmark_type", "base")
        question = sample.get("question", "")
        tables_blob = sample.get("tables", "")
        correct_answer = sample.get("correct_answer") or sample.get("correct answer")
        qtype = sample.get("qtype", "unknown")
        scale = sample.get("scale", "unknown")

        overall["total"] += 1
        by_qtype[qtype]["total"] += 1
        by_scale[scale]["total"] += 1

        prompt = build_question_prompt(question, tables_blob)

        record = {
            "id": sample.get("id"),
            "scale": scale,
            "qtype": qtype,
            "question": question,
            "correct_answer": correct_answer,
            "benchmark_type": benchmark_type,
            "llm_answer": None,
            "judge_response": None,
            "evaluation_result": None,
            "model": model,
            "judge_model": judge_model,
        }

        try:
            answer = call_model(prompt, model, use_openai)
            record["llm_answer"] = answer
        except Exception as exc:
            record["evaluation_result"] = "llm_error"
            record["error"] = str(exc)
            overall["evaluation_failed"] += 1

        records.append(record)

    # Phase 2: Judge collected answers
    records_to_judge = [r for r in records if r["evaluation_result"] is None]

    for record in tqdm(records_to_judge, desc="Judging answers"):
        question = record["question"]
        answer = record["llm_answer"]
        benchmark_type = record["benchmark_type"]
        correct_answer = record["correct_answer"]
        qtype = record["qtype"]
        scale = record["scale"]

        judge_prompt = build_judge_prompt(question, answer, str(correct_answer), benchmark_type)

        try:
            judge_response = call_model(judge_prompt, judge_model, judge_use_openai)
            record["judge_response"] = judge_response
        except Exception as exc:
            record["evaluation_result"] = "judge_error"
            record["error"] = str(exc)
            overall["evaluation_failed"] += 1
            continue

        decision = extract_judge_decision(record["judge_response"])
        if decision is None:
            record["evaluation_result"] = "evaluation_failed"
            overall["evaluation_failed"] += 1
        elif decision == "yes":
            record["evaluation_result"] = "yes"
            overall["correct"] += 1
            by_qtype[qtype]["correct"] += 1
            by_scale[scale]["correct"] += 1
        else:
            record["evaluation_result"] = "no"
            overall["incorrect"] += 1

    with output_path.open("w", encoding="utf-8") as results_file:
        for record in records:
            results_file.write(json.dumps(record) + "\n")

    summary_payload = {
        "model": model,
        "judge_model": judge_model,
        "overall": {
            **overall,
            "accuracy": (overall["correct"] / overall["total"] * 100) if overall["total"] else 0.0,
        },
        "by_qtype": aggregate_counts(by_qtype),
        "by_scale": aggregate_counts(by_scale),
        "total_samples": len(samples),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simplified table benchmark using a Hugging Face dataset.")
    parser.add_argument("--model", required=True, help="LLM model name for answering questions.")
    parser.add_argument(
        "--output",
        default="simple_results.jsonl",
        help="Where to store per-sample benchmark outputs.",
    )
    parser.add_argument(
        "--summary",
        default="simple_results_summary.json",
        help="Where to store aggregated metrics.",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen2.5:32b",
        help="LLM model name used as the judge.",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Force the answering model to use the OpenAI client.",
    )
    parser.add_argument(
        "--judge-use-openai",
        action="store_true",
        help="Force the judge model to use the OpenAI client.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples processed from the input file.",
    )
    parser.add_argument(
        "--hf-dataset",
        default="trl-lab/tabular-reasoning",
        help="Hugging Face dataset identifier to load.",
    )
    parser.add_argument(
        "--hf-split",
        default="test",
        help="Which split to load from the Hugging Face dataset (default: test).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token to access private datasets (passed to datasets.load_dataset via use_auth_token).",
    )
    parser.add_argument(
        "--share",
        type=parse_share,
        default=1.0,
        help="Fraction of samples per (scale, qtype) to evaluate (0-1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    run_simple_benchmark(
        output_path=output_path,
        summary_path=summary_path,
        model=args.model,
        judge_model=args.judge_model,
        force_openai=args.use_openai,
        force_judge_openai=args.judge_use_openai,
        max_samples=args.max_samples,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_token=args.hf_token,
        share=args.share,
    )


if __name__ == "__main__":
    main()
