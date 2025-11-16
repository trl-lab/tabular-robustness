# How well do LLMs reason over tabular data, really?

This repository contains the benchmark suite and replication package for our paper "How well do LLMs reason over tabular data, really?", presented at the 4th Table Representation Learning Workshop at ACL 2025. It lets you reproduce the reasoning tests from the paper and explore how different models perform on table reasoning challenges.

[Read the paper](https://www.alphaxiv.org/abs/2505.07453) for a deeper discussion of the results.

## Simple benchmark runner

The most accessible way to rerun the reasoning test from the paper is `simple_benchmark.py`. It loads the Hugging Face table datasets, builds prompts that mirror our published evaluation, and drives both question answering and judgement either through vLLM (preferred) or OpenAI when you want to avoid local GPU requirements.

When you run with vLLM the script needs a CUDA-enabled GPU with enough memory to host the chosen model (qwen2.5:32b needs ≳20 GB of VRAM). If you don't have a compatible GPU you can still use the OpenAI client path, but you lose the reproducibility benefits of running fully on-device.

```bash
python simple_benchmark.py \
  --model qwen2.5:32b \
  --output simple_results.jsonl \
  --summary simple_results_summary.json \
  --judge-model qwen2.5:32b \
```

This script:
1. Loads the [`trl-lab/tabular-reasoning`](https://huggingface.co/datasets/trl-lab/tabular-reasoning) dataset from Hugging Face.
2. Renders every record into a prompt with tables, questions, and step-by-step instructions.
3. Calls vLLM (or OpenAI) to answer and to judge each response using the same evaluation logic as the paper.
4. Emits per-sample JSONL records plus an accuracy summary similar to what we reported.

If you want to dive into the exact dataset loaders, aggregation logic, or LaTeX exports that powered the paper you can explore the `src/` package. Those `run_*` scripts orchestrate the base/missing/shuffle benchmarks at multiple scales and represent the “more in-depth” code behind the published numbers. For a reference to every CLI option you can pass to `simple_benchmark.py`, see [SIMPLE_BENCHMARK_ARGS.md](./SIMPLE_BENCHMARK_ARGS.md).

## Project layout

```
tabular-robustness/
├── simple_benchmark.py   # Lightweight runner for the reasoning benchmark (uses vLLM/OpenAI).
├── tabreasbench/         # Core CLI helpers, evaluation scripts, and the `run_*` suite.
│   ├── data/             # Local copy of the datasets and scalings used during benchmarking.
│   ├── src/              # Full benchmark orchestration, dataset parsing, aggregation, and LaTeX exports behind the paper’s reports.
│   ├── cli.py
│   ├── run_benchmark.py
│   └── run_full_benchmark.py
├── pyproject.toml
├── requirements.txt     # Lists the runtime dependencies used by `simple_benchmark.py`.
├── README.md            # This document.
└── LICENSE
```

`simple_benchmark.py` sits at the repository root for quick experimentation, while `tabreasbench/src/` contains the more in-depth benchmark code referenced in the paper.

## Getting started

1. Clone the repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your GPU has enough VRAM for the vLLM model you want to run.
4. Run `simple_benchmark.py` (shown above) or explore the `src/` roster of `run_*` scripts for full benchmark sweeps.

## Citation

If you use our test code in your research, please cite our paper:

```bibtex
@inproceedings{wolff2025well,
  title={How well do LLMs reason over tabular data, really?},
  author={Wolff, Cornelius and Hulsebos, Madelon},
  booktitle={The 4th Table Representation Learning Workshop at ACL 2025}
}
```

Plain text citation:
```
Wolff, C., & Hulsebos, M. How well do LLMs reason over tabular data, really?. In The 4th Table Representation Learning Workshop at ACL 2025.
```
