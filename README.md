<div align="center">
<h1 align="center">Effects of structure on reasoning in instance-level Self-Discover</h1>
<p align="center">
    <em>An open-source implementation for the research paper investigating the effects of structured versus unstructured reasoning in Large Language Models using instance-level Self-Discover.</em>
</p>
<p align="center">
    <a href="[LINK_TO_YOUR_PAPER_WHEN_PUBLIC_OR_PREPRINT]"><img src="https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader" alt="Paper PDF"></a>
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=plastic&logo=python" alt="Python Version">
    <a href="https://github.com/sachith-gunasekara/self-discover/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=plastic" alt="License"></a>
    <a href="https://python-poetry.org/"><img src="https://img.shields.io/badge/Poetry-Package%20Manager-purple?style=plastic&logo=poetry" alt="Poetry"></a>
</p>
</div>

---

## 📖 About The Project

This repository contains the code and experimental setup for the research paper titled "Effects of structure on reasoning in instance-level Self-Discover". The project introduces **iSELF-DISCOVER**, an instance-level adaptation of the SELF-DISCOVER framework, to empirically evaluate the performance of dynamically generated structured JSON reasoning against its unstructured, natural language counterpart.

Our findings, particularly on benchmarks like MATH, BBH, and a replicated T4D, suggest a consistent advantage for unstructured reasoning plans. This work aims to provide insights into optimal plan generation granularity (instance-level vs. task-level) and the nuanced reliance on structured formats for complex LLM problem-solving.

---

## 🚀 Key Features

- Implementation of the **iSELF-DISCOVER** framework.
- Support for both **structured (JSON)** and **unstructured (natural language)** reasoning plan generation and execution.
- Evaluation scripts for benchmarks: **BBH, T4D (replicated), and MATH**.
- Integration with models like **LLaMA-3.1-405B-Instruct** and **Mistral-Large** via APIs.
- Configuration for 0-shot and few-shot (e.g., 5-shot) guidance for plan generation.

---

## 🛠️ Getting Started

### Prerequisites

- Python (e.g., 3.9+)
- Poetry (for dependency management)
- API Keys:
  - `MISTRAL_API_KEY`
  - `LAMBDA_LABS_API_KEY`

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sachith-gunasekara/self-discover.git
    cd self-discover
    ```

2.  **Set up your Python virtual environment:**
    It's recommended to create and activate a virtual environment. For example:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies using Poetry:**

    ```bash
    poetry install
    ```

4.  **Set up API Keys:**
    Create a `.env` file in the root of the project with your API keys:
    ```env
    MISTRAL_API_KEY="YOUR_MISTRAL_API_KEY"
    LAMBDA_LABS_API_KEY="YOUR_LAMBDA_LABS_API_KEY"
    ```

---

## ⚙️ Configuration

Experiments are primarily configured via `evals/config.toml`.

Key parameters in `evals/config.toml`:

```toml
[MODEL]
# Specifies the model family to use.
# "mistral" will use "mistral-large-2407".
# "llama" will use "llama3.1-405b-instruct-fp8".
model_type = "mistral"

[EVAL]
# batch_size controls the number of instances processed in a single batch.
# This is useful for managing API rate limits or resource usage.
batch_size = 5
# wait_time is the duration (in seconds) to wait if any Exceptions arises during the evaluation process (Can allow breathing room if unexpected API errors occur).
wait_time = 1
```

---

## 🔬 Running Research Experiments

To reproduce the experiments presented in the paper:

1.  **Ensure your Poetry environment is active:**
    If you haven't already, activate it:

    ```bash
    source .venv/bin/activate # Or your chosen environment activation command
    # Alternatively, you can prefix commands with `poetry run`
    ```

2.  **Prepare Log Directory:**
    The evaluation scripts are designed to check for current progress in `evals/logs` and resume. If you want to run experiments from scratch, **ensure the `evals/logs` directory is deleted or empty.**

3.  **Run evaluation scripts (from the project root directory):**

    - **To evaluate the iSELF-DISCOVER approach (our proposed method):**
      Use `evals/iself_discover_eval.py`. Key arguments include:

      - `--structured`: (Flag, no value) Use structured JSON reasoning. If omitted, defaults to unstructured.
      - `--few_shot_examples <N>`: Number of few-shot examples to use (e.g., `0` for zero-shot, `5` for five-shot). Defaults to `0`.
      - `--stream`: (Flag, no value) Stream LangGraph steps one by one and log debug messages for each output from every stage. Defaults to `False`.

      Example (unstructured, 0-shot, run from root):
      ```bash
      python evals/iself_discover_eval.py
      ```
      Example (structured, 5-shot, with streaming, run from root):
      ```bash
      python evals/iself_discover_eval.py --structured --few_shot_examples 5 --stream
      ```

    - **To evaluate the original SELF-DISCOVER approach (baseline):**
      Use `evals/self_discover_eval.py`. Key arguments include:

      - `--phase <PHASE_VALUE>`: **(Required for research experiments)** Specify the stage of the SELF-DISCOVER framework to run.
        - Use `1` to run Phase I (e.g., task-specific reasoning structure discovery).
        - Use `2` to run Phase II (e.g., solving instances using a discovered structure).
          While the script might default to running both if this argument is omitted, for research evaluation purposes, **you must explicitly specify either `1` or `2`**.
      - `--stream`: (Flag, no value) Stream LangGraph steps and log debug messages. Defaults to `False`.

      Example (running Phase I, no streaming, run from root):
      ```bash
      python evals/self_discover_eval.py --phase 1
      ```
      Example (running Phase II with streaming, run from root):
      ```bash
      python evals/self_discover_eval.py --phase 2 --stream
      ```

### Datasets

The necessary datasets (BBH, replicated T4D, MATH subsample) are expected to be handled by the scripts, potentially by downloading from sources mentioned in the paper (e.g., Hugging Face `sachithgunasekara/framework-MATH-subsample` and `sachithgunasekara/t4d`). No manual dataset placement should be required.

### Expected Output

Experimental results, detailed logs, and any generated reasoning traces will be stored in the `evals/logs` directory. The structure within this directory should allow for identification of results based on the benchmark, model, and experimental configuration.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` file for more information.

---

## 🙏 Acknowledgements

- Authors of the original SELF-DISCOVER paper (Zhou et al., 2024).
- Mistral AI for providing a generous free-tier access.
- The open-source community for tools and libraries that made this work possible.

---
