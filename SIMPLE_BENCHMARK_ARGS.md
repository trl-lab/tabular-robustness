# `simple_benchmark.py` CLI reference

This document describes every `argparse` option exposed by `simple_benchmark.py`, including what it controls and the default behavior.

| Option | Description | Default |
| --- | --- | --- |
| `--model` | Model name or path to use for answering the questions (e.g., `qwen2.5:32b`). Required. | (none) |
| `--output` | Path for the per-sample JSONL results file. | `simple_results.jsonl` |
| `--summary` | Path for the aggregated summary JSON containing accuracy metrics. | `simple_results_summary.json` |
| `--judge-model` | Model name/path used for the judge prompt that votes on correctness. Defaults to `qwen2.5:32b`. | `qwen2.5:32b` |
| `--use-openai` | Flag that forces the answering model to use the OpenAI API instead of local vLLM. | disabled |
| `--judge-use-openai` | Flag that forces the judge model to use OpenAI instead of vLLM. | disabled |
| `--max-samples` | Integer limit to process only the first `n` records from the dataset. Useful for quick checks. | no limit (`None`) |
| `--hf-dataset` | Hugging Face dataset identifier to load (must contain the expected schema). | `trl-lab/tabular-reasoning` |
| `--hf-split` | Which split to use from the dataset (e.g., `train` or `test`). | `test` |
| `--hf-token` | Optional Hugging Face token for accessing private datasets. | not set (`None`) |
| `--share` | Fraction of samples to keep per `(scale, qtype)` bucket. Must be between 0 and 1. | `1.0` |
| `--vllm-batch-size` | Number of prompts sent per vLLM batch. Lower values reduce GPU memory usage. | `16` |
| `--vllm-world-size` | Optional tensor-parallel world size passed to vLLM for multi-GPU runs. | not set (`None`) |

The script enforces `0 <= share <= 1` through `parse_share`. Use the `--use-openai` and `--judge-use-openai` flags whenever you want to avoid GPU requirements for the answering or judge models, respectively.
