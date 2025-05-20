# Overview

This repository stores the code used for the experiments in the "Incorporating Token Usage into Prompting Strategy Evaluation" paper. Unfortunately, the official [LM Evaluation Harness repository](https://github.com/EleutherAI/lm-evaluation-harness) had some breaking bugs that we had to fix prior to running our experiments. We were unable to merge all of those fixes prior to submission and thus provide this fork for reproducing our results.

# Reproducing the Results

To reproduce the results from our experiments, we provide accompanying `*.py` files. We use these Python files instead of the commandline tools LM Evaluation Harness provides because it allows greater access to parameters and we want each combination of model, prompting strategy, and benchmark to be individually replicable.

## Setup

### Environment

We use `uv` for package management, which can be installed via [the instructions found here](https://docs.astral.sh/uv/getting-started/installation/). Once `uv` is installed, the virtual environment can be prepared, from the **project root**, as follows:
```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[vllm]"
source .venv/bin/activate
```

Enter the `lm_eval` directory from the **project root** to be able to run the experiments:
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

After receiving model access, login to the Hugging Face CLI and follow the instructions output from the following command to access the models locally:
```bash
huggingface-cli login
```

## Execution

Within the `lm_eval` directory, there are a number of `<model-benchmark-strategy>.py` files. These contain, along with the task files from the `tasks` directory, define the exact configurations and hyperparameters for reproducing our results. Note that we use LM Evaluation Harness in this way (as opposed to the CLI) to (1) have access to more variables, (2) allow users to reproduce individual experiments, and (3) allow each experiment to be examined in isolation.

To reproduce the results, run the desired Python file:

```bash
python <experiment-name>.py
```

> Note that the values for `tensor_parallel_size`, `data_parallel_size`, and `batch_size` may need to be adjusted according to the GPU setup and memory constraints
  
The results will be stored in the newly created `results` directory in `lm_eval`.

# Results

## Retrieving the Data

The results of our experiments are available in the `results.tar.gz` file available at this link <link redacted; sample included in supplementary materials>. This file contains the results of all experiments, including the ablation studies and general results.
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