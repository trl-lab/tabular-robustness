#!/usr/bin/env python3
"""
Push a JSONL file into a Hugging Face Datasets repository.

Usage:
  python push_to_hf.py \
    --jsonl /path/to/data.jsonl \
    --repo-id username/my-dataset \
    --split-name train \
    --private \
    --commit-message "Initial upload"

Auth:
  - Set an environment variable HF_TOKEN with a write-access token, or
  - Pass --token YOUR_TOKEN

Notes:
  - Requires: datasets>=2.14.0, huggingface_hub>=0.23.0
  - If the repo doesn't exist, it will be created.
"""
import argparse
import os
import sys
from typing import List, Optional

from datasets import load_dataset, Dataset, DatasetDict  # type: ignore
from huggingface_hub import HfApi  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push JSONL to a Hugging Face dataset repo")
    p.add_argument(
        "--jsonl",
        nargs=+1,
        required=True,
        help="Path(s) to JSONL file(s). You can pass multiple to concatenate.",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="Target dataset repo like 'username/dataset_name'",
    )
    p.add_argument(
        "--split-name",
        default="train",
        help="Dataset split name to assign (default: train)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it doesn't exist",
    )
    p.add_argument(
        "--commit-message",
        default="Add dataset",
        help="Commit message for the upload",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF API token (defaults to HF_TOKEN env var)",
    )
    p.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Max shard size used by push_to_hub (e.g., '500MB', '1GB')",
    )
    return p.parse_args()


def ensure_repo(repo_id: str, token: Optional[str], private: bool) -> None:
    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
        print(f"✔️  Ensured dataset repo exists: {repo_id} (private={private})")
    except Exception as e:
        print(f"❌ Failed to ensure/create repo '{repo_id}': {e}")
        sys.exit(1)


def load_jsonl_as_dataset(files: List[str]) -> Dataset:
    # datasets will auto-detect JSON Lines when using the 'json' builder
    print(f"📦 Loading JSONL: {files}")
    ds = load_dataset("json", data_files=files, split="train")  # type: ignore
    # Basic sanity check
    print(f"✅ Loaded {len(ds):,} rows with columns: {list(ds.features.keys())}")
    return ds


def push_dataset(
    ds: Dataset,
    repo_id: str,
    split_name: str,
    token: Optional[str],
    commit_message: str,
    max_shard_size: str,
) -> None:
    print(
        f"🚀 Pushing split='{split_name}' to https://huggingface.co/datasets/{repo_id} ..."
    )
    # Push a single split; this will create a DatasetDict with that split on the Hub
    ds.push_to_hub(
        repo_id=repo_id,
        split=split_name,
        token=token,
        commit_message=commit_message,
        max_shard_size=max_shard_size,
    )
    print("🎉 Upload complete!")


def main() -> None:
    args = parse_args()

    if not args.token:
        print(
            "❌ No token provided. Set HF_TOKEN env var or pass --token.\n"
            "   Create a token at https://huggingface.co/settings/tokens"
        )
        sys.exit(2)

    # Expand paths and verify they exist
    files = []
    for f in args.jsonl:
        f = os.path.expanduser(f)
        if not os.path.isfile(f):
            print(f"❌ File not found: {f}")
            sys.exit(2)
        files.append(f)

    ensure_repo(args.repo_id, args.token, args.private)

    ds = load_jsonl_as_dataset(files)

    # Optional: cast to string for problematic columns (commented; uncomment if needed)
    # from datasets import Features, Value
    # ds = ds.cast(Features({k: Value("string") for k in ds.features}))

    push_dataset(
        ds=ds,
        repo_id=args.repo_id,
        split_name=args.split_name,
        token=args.token,
        commit_message=args.commit_message,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
