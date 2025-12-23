# Research Summary: Airflow DAG Generation with SMLs

This document summarizes the research conducted to fine-tune a Small Language Model (SML) for generating Apache Airflow DAGs. It covers the findings, lessons learned, and the technical details of the fine-tuning process.

## 1. Research Conducted & Findings

The goal of this research was to determine if a specialized SML (specifically **Qwen 2.5 1.5B Instruct**) could be fine-tuned to generate high-quality, valid Airflow DAGs from natural language instructions, running efficiently on consumer hardware (Mac M1/M2).

### The Data Pipeline
We established a robust three-stage data pipeline:
1.  **Mining**: Extracted raw Python DAG files from the official Apache Airflow repository across multiple versions. Identifying "valid" DAGs required AST parsing to filter out non-DAG code.
2.  **Instruction Generation**: Used **Anthropic's Claude 3.5 Sonnet** via the **Batch API** to generate synthetic user instructions for each DAG.
    *   **Finding**: The Batch API proved extremely cost-effective. We generated a comprehensive dataset for **under $2**, proving that synthetic data generation is accessible for individual researchers.
3.  **Dataset Creation**: Cleaned and formatted the pairs into ChatML format, creating the `airflow-dag-dataset`.

### Inference Optimization
We tackled inference in two distinct contexts:

1.  **Research & Evaluation (Cloud GPU)**:
    *   **Unsloth**: For generating the large test sets needed for evaluation (thousands of DAGs), we relied on **Unsloth** running on NVIDIA GPUs (Colab). Its specific optimizations for standard GPUs provided a **2x speedup** over standard Hugging Face inference, making the iterative research process viable.

2.  **Production Deployment (Local Mac)**:
    *   **Initial Attempts**: Local inference using `mlx` and standard `transformers` was functional but often compute-bound or memory-hungry.
    *   **Llama.cpp C++ Server**: Offered true parallelism and continuous batching, significantly increasing throughput and utilizing the GPU well. However, it introduced complexity in deployment.
    *   **Winner**: We settled on the **Python `llama.cpp` server wrapper**. It provided a sweet spot of "Library First" simplicity effectively zero performance loss compared to the raw C++ backend, while simplifying the architecture into a unified CLI (`airflow-net serve`).

### Model Performance
*   **Validity**: The fine-tuned model generated **8% fewer invalid DAGs** compared to the base model.
*   **Syntax**: It successfully learned modern Airflow syntax, avoiding deprecated patterns common in older base models.
*   **Hallucinations**: The model occasionally learned to use internal Airflow testing libraries (e.g., for unit tests) because these files were present in the mined training data. This highlights the critical importance of data filtering.

## 2. Lessons Learned

### "Garbage In, Garbage Out" applies to Code
The most distinct issue encountered was the model "hallucinating" internal testing methods.
*   **Cause**: The mining script picked up `example_*.py` files that were actually internal tests, not user-facing DAGs.
*   **Lesson**: Rigorous filtering of training data is more valuable than larger model sizes. We need to exclude test files and ensure the "gold standard" code is purely user-centric.

### Hardware Trade-offs
*   **Training**: Training on Mac M1 (MLX) was too slow (~20 hours for 3 epochs). Cloud GPUs are necessary for this step. An **NVIDIA A100** on Colab reduced training time to **~30 minutes**.
*   **Inference**: Mac M1 is memory-bandwidth limited. Running multiple small models in parallel helps saturate the GPU better than a single stream, but continuous batching is the key to maximizing token throughput.

### Simplicity Wins
We initially over-engineered the inference stack with C++ servers and complex concurrency. Reverting to a Python-centric approach using `llama.cpp`'s Python bindings essentially matched the performance while drastically reducing code complexity and maintenance burden.

## 3. How Fine-Tuning Works

We used **Unsloth** for efficient QLoRA fine-tuning, which allows fitting the training process into a free Google Colab T4 instance (or faster on an A100).

### Technical Configuration

| Parameter | Value | Reasoning |
| :--- | :--- | :--- |
| **Model** | `Qwen/Qwen2.5-1.5B-Instruct` | Excellent coding capabilities in a small footprint. |
| **Method** | **QLoRA** (4-bit) | Reduces memory usage by ~50% allowing large context windows. |
| **Context Window** | **4096 tokens** | Sufficient for most complex DAG files. |
| **LoRA Rank (r)** | 16 | Standard balance between adaptability and parameter efficiency. |
| **LoRA Alpha** | 16 | 1:1 scaling with rank. |
| **Learning Rate** | `2e-4` | Standard for QLoRA fine-tuning. |
| **Batch Size** | 2 (Per device) | Limited by GPU VRAM. |
| **Grad. Accumulation** | 4 | Effective batch size of 8. |
| **Target Modules** | All linear layers | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. |

### The Training Process
1.  **Preparation**: The dataset is loaded and formatted with the `Qwen` chat template (`<|im_start|>user...`).
2.  **Training**: The model trains for **1 epoch** (or ~60 steps) on the dataset. We found that even short training runs significantly adapted the style.
3.  **Merging**: The LoRA adapters (small diff files) are merged back into the base model weights.
4.  **Quantization**: The final model is converted to **GGUF format** (typically `q4_k_m`) for efficient local inference on the user's machine.
