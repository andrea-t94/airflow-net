# From Toy to Tool: Fine-Tuning an SLM for Airflow DAGs (Part 1)

Hey, I'm Andrea. I'm currently working as an MLE at Flix, coming from a classical ML background where I work on customer personalization—churn models, causal ML applied to CLV, that sort of thing. I've always loved deep learning and AI; I contributed to some DL/AI projects in the past and now I want to deep dive into the topic given how important it's become.

When I started this project, I expected fine-tuning language models to feel completely foreign. Honestly? It didn't. The loop is the same one I've been running for years: collect data, train, evaluate, find problems, fix the data, train again. The tools have different names—LoRA instead of regularization, tokenizers instead of feature encoders—but the muscle memory transfers.

I couldn't find end-to-end content that treated SLMs as engineering artifacts rather than magic. Most stuff out there is either pure theory (here's how attention works!) or toy tutorials (run this Colab notebook!). Neither helps when you're trying to actually ship something. So I'm documenting what I learned building AirflowNet—a small language model specialized in writing Airflow DAGs. The failures turned out to be more interesting than the successes.

This is a three-part series:

**Part 1 (this post):** Data collection, fine-tuning, and evaluation. I'll walk through how I built the training dataset, what choices I made, and how those choices came back to bite me. The model learned to hallucinate test utilities because I forgot to filter them out of my training data. Classic.

**Part 2:** Local deployment. Getting the model running at 65 tokens/second on a Mac M1, why inference optimization is completely different from training, and how I integrated it into a CLI and MCP server.

**Part 3:** Reflections and roadmap. A honest look at the problems I faced throughout this project, what I'd do differently, and—hopefully with some feedback from readers—a roadmap of things to try next.

---

## The TLDR

I fine-tuned a 1.5B parameter model (Qwen 2.5 Coder) on ~10k Airflow DAG examples. The results were mixed in interesting ways. Idiomatic Airflow usage improved 4x over baseline—the model learned to use proper operators like `SnowflakeOperator` instead of wrapping everything in `PythonOperator`. But hallucinations actually got worse: the model confidently generates imports like `from tests_common.test_utils.system_tests import get_test_run` because that pattern was in my training data. It also learned that all numbers should be "20" because 85% of my synthetic instructions used that as a placeholder. The whole experience reinforced something I already knew from classical ML: your model is only as good as your data. With SLMs, you just discover this in more creative ways.