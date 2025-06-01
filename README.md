# How well do LLMs reason over tabular data, really?

This repository contains the benchmark suite and replication package for our paper "How well do LLMs reason over tabular data, really?". It allows you to reproduce our benchmark results and evaluate language models on their table reasoning capabilities.

[Read our paper](https://arxiv.org/abs/2402.19427) for a detailed analysis of the benchmark results and findings.

## Testing models

To replicate the results from our paper or test new available models, follow these steps:

### Prerequisites

Before installing our TabReasBench package, you need to have Ollama installed and the required model pulled:

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull the required model:
```bash
ollama pull qwen2.5:32b
```

### Hardware Requirements

- For running the benchmarks: Any GPU that can run Ollama models
- For evaluating results: GPU with at least 20GB VRAM (required for qwen2.5:32b used as LLM-as-a-judge)

### Installation

You can install TabReasBench using pip:

```bash
pip install tabreasbench
```

### Running the Benchmarks

To replicate our benchmark results, run:

```bash
tabreasbench --model qwen2.5:32b --output_dir benchmark_results
```

This will:
1. Run all benchmarks (base, missing, and shuffle) across different scales
2. Evaluate the results using qwen2.5:32b as the judge model
3. Generate aggregated results and LaTeX tables

The results will be organized in the specified output directory:
```
output_dir/
├── raw_results/
│   └── results_qwen2.5_32b.csv           # Raw model outputs and ground truth
├── evaluated_results/
│   └── results_qwen2.5_32b_evaluated.csv # Results with correctness evaluation
└── aggregated_results/
    ├── overall_summary.csv                # Overall performance metrics
    ├── detailed_results.csv               # Per-dataset performance breakdown
    ├── overall_summary.tex                # LaTeX table of overall results
    └── detailed_results.tex               # LaTeX table of detailed results
```

## Citation

If you use our test code in your research, please cite our paper:

```bibtex
@article{wolff2025well,
  title={How well do LLMs reason over tabular data, really?},
  author={Wolff, Cornelius and Hulsebos, Madelon},
  journal={arXiv preprint arXiv:2505.07453},
  year={2025}
}
```

Plain text citation:
```
Wolff, C., & Hulsebos, M. (2025). How well do LLMs reason over tabular data, really?. arXiv preprint arXiv:2505.07453.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.