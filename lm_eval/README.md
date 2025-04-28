# Overview

This repository stores the code used for the experiments in the "Incorporating Token Usage into Prompting Strategy Evaluation" paper. Unfortunately, the official [LM Evaluation Harness repository](https://github.com/EleutherAI/lm-evaluation-harness) had some breaking bugs that we had to fix prior to running our experiments. We were unable to merge all of those fixes prior to submission and thus provide this fork for reproducing our results.

# Reproducing the Results

To reproduce the `*.json` files found in the `results` directory, we provide accompanying `*.py` files. We use these Python files instead of the commandline tools LM Evaluation Harness provides because it allows greater access to parameters and we want each combination of model, prompting strategy, and benchmark to be individually replicable.

## Setup

### Environment

We use `uv` for package management, which can be installed via [the instructions found here](https://docs.astral.sh/uv/getting-started/installation/). Once `uv` is installed, the virtual environment can be prepared as follows:
```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[vllm]"
source .venv/bin/activate
```
from the **root** of this repository.

Enter the `lm_eval` directory from the project root to be able to run the experiments:
```bash
cd lm_eval
```

Set the environment variable to allow full context window usage:
```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

### Model Access

We use Llama 3.1 8B Instruct and Qwen 2.5 14 and 32B Instruct models, which have gated access. To access these models, you will need to have an account on the Hugging Face model hub and review and accept the terms of service for each:
* [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
* [Qwen 2.5 14B Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
* [Qwen 2.5 32B Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)

After receiving model access, login to the Hugging Face CLI to access the models locally:
```bash
huggingface-cli login
```

## Execution

The following instructions should be performed in the `lm_eval` directory to reproduce our results:

* Run the Python file: `python <experiment-name>.py`
  * Note that the values for `tensor_parallel_size`, `data_parallel_size`, and `batch_size` may need to be adjusted according to the GPU setup and memory constraints
  
The results will be stored in the newly created `results` directory in `lm_eval`.

# Results

## Retrieving the Data

The results of our experiments are available in the `results.tar.gz` file available at this link (redacted). This file contains the results of all experiments, including the ablation studies and general results.
Once downloaded, it can be decompressed with:
```bash
tar -xvzf path/to/results.tar.gz
```

## Structure

Results from our decompressed experiments are stored in the `results` directory as follows:
```text
results/
    ├── ablation/
    |   ├── <prompting-strategy>/
    |   │   ├── <benchmark>/
    |   │   │   ├── <model>/
    |   │   │   │   ├── <num-fewshot>.json
    |   │   │   │   ├── <num-fewshot>.json
    |   │   │   │   ├── ...
    |   │   │   ├── ...
    |   │   ├── ...
    ├── general/
        ├── <prompting-strategy>/
        │   ├── <benchmark>/
        │   │   ├── <model>/
        │   │   │   ├── <experiment-name>.json
        │   │   ├── ...
        │   ├── ...
```
The `*.json` files contain the results of the experiments exactly as they are produced by LM Evaluation Harness. We rely on their established format for consistency.