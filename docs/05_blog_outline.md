# Blog Series: From Toy to Tool – Fine-Tuning an SLM for Airflow DAGs

**Series Structure:** 2 parts, ~2500-3000 words each
**Target Audience:** Data/ML engineers with solid Python and classical ML background who want to understand the end-to-end SLM fine-tuning process

---

## Part 1: Fine-Tuning an Airflow SLM (And Why It Hallucinated More Than the Baseline)

**Core Promise:** Show the complete fine-tuning loop—data collection through evaluation—while being honest about what worked and what didn't.

**Opening Hook (2-3 paragraphs):**

Don't start with "who I am." Start with the punchline that makes readers want to continue.

Opening should be something like: *"I fine-tuned a 1.5B parameter model to write Airflow DAGs. It learned to use the right operators 4x more often than the baseline. It also hallucinated internal test utilities that don't exist in production Airflow. Both of these results came from the same training data."*

This immediately establishes: (1) you did real work, (2) you have specific numbers, (3) the story is nuanced, not a victory lap.

Follow with a brief context setter: *"I'm a senior MLE at Flix with a classical ML background. When I started this project, I expected fine-tuning language models to feel completely foreign. It didn't. The process is remarkably similar to what we already do—collect data, train, evaluate, identify issues, iterate. The tools have different names, but the loop is the same."*

Then state what the post covers: the data strategy, training setup, and evaluation pipeline—with emphasis on the failures and what they taught you.

---

### Section 1: The Evaluation Results (Start Here, Not at the End)

**Why lead with results:** Readers want to know if this is worth their time. Showing the outcomes first hooks them, then they'll want to understand how you got there.

**Content to include:**

Present the comparison table with actual numbers from your evaluation:

| Metric | Baseline (Qwen 2.5 1.5B) | Fine-tuned | What This Means |
|--------|--------------------------|------------|-----------------|
| Syntax Validity | ~X% | ~X+8% | Fewer broken DAGs |
| Idiomatic Airflow | 11% | 43% | Uses proper operators instead of PythonOperator wrappers |
| Hallucination-free | 24% | 6% | **Worse**—learned test utilities from training data |
| Instruction Following | 15% | 8% | **Worse**—overfitted to synthetic patterns |

**Key narrative beats:**

1. The good: 4x improvement in idiomatic usage means the model learned what Airflow-specific code looks like. It uses `SnowflakeOperator` instead of wrapping everything in `PythonOperator` with hooks.

2. The bad (and interesting): The hallucination rate got *worse*. Root cause analysis revealed two problems:
   - Training data included files from `tests_common/` directory—internal Airflow CI/CD utilities that don't exist in production
   - The model confidently generates `from tests_common.test_utils.system_tests import get_test_run` because it saw this pattern repeatedly

3. The ugly: Instruction following degraded because 85.7% of training examples used "20" as a dummy number ("insert 20 records," "20 retries," "20 seconds"). The model learned that all numbers should be 20.

**Transition:** "These results didn't come from nowhere. They came from specific choices in data collection and preprocessing. Let me walk through what I did and why these failures were predictable in hindsight."

---

### Section 2: Data Strategy—Where the Problems Started

**The Setup (1-2 paragraphs):**

Explain the core difference from classical ML: *"In classical ML, I spent weeks engineering features—rolling averages, interaction terms, careful normalization. With SLMs, the model extracts its own features. Our job shifts from feature engineering to data curation. The model will learn whatever patterns exist in your data, including patterns you didn't intend to teach it."*

**Dataset Composition (~10k samples):**

Present this as prose, not a bulleted breakdown:

*"I built the dataset from two sources. About 65% (6,500 samples) came from the official Apache Airflow repository—DAG files that represent best practices. The remaining 35% came from the Magpie dataset, general Python code meant to prevent catastrophic forgetting of basic syntax. The ratio was somewhat arbitrary; I wanted enough Airflow-specific examples to specialize the model without losing general Python competence."*

**The Instruction Generation Process:**

This is where you explain how you created instruction-response pairs:

*"Raw code isn't a training dataset. You need (instruction, response) pairs. I used Claude's Batch API to generate three different instructions per DAG file—variations on 'write a DAG that does X.' The Batch API was a revelation: generating 10k instructions cost under $2, compared to $5+ with synchronous requests. No latency requirements meant I could wait for batch processing."*

**Preprocessing Decisions (and their consequences):**

Walk through what you did and connect it to the evaluation failures:

1. **Stripped comments and docstrings:** "I removed copyright headers and verbose docstrings to reduce noise. The model should learn logic, not boilerplate. This worked as intended."

2. **Didn't filter test files:** "I scraped the entire Airflow repo without excluding the `tests/` and `tests_common/` directories. This seemed fine at the time—more data is better, right? Wrong. The model learned to import `tests_common.test_utils.system_tests` because those files were in the training set. Classic garbage-in, garbage-out."

3. **Synthetic instruction homogeneity:** "When Claude generated instructions, it defaulted to using '20' as a placeholder number constantly. I didn't catch this during data inspection. The model learned that '20' is the universal correct answer for any numerical parameter."

**Show a concrete before/after example:**

Include an actual diff or side-by-side comparison of a raw DAG file vs. the cleaned version. This makes the preprocessing tangible.

**The Lesson (explicit):**

*"Data preprocessing for SLMs isn't about feature scaling or encoding categoricals. It's about understanding what patterns you're implicitly teaching. Every file in your training set is a lesson. Every repeated pattern becomes a learned behavior."*

---

### Section 3: The Training Setup

**Hardware Reality (1 paragraph):**

*"I don't own GPUs. I used Google Colab Pro with an A100. The free T4 tier works but training takes 3-4x longer. For a 1.5B model on 10k samples, A100 training completed in about 40 minutes. On T4, expect 2-3 hours."*

**The Optimization Stack (explain why, not just what):**

Present this as a natural explanation, not a feature list:

*"Training transformers is expensive—memory scales quadratically with sequence length. A 1.5B model with 2048 context would normally require 40GB+ VRAM. I used two techniques to make this feasible on Colab:"*

1. **QLoRA:** "Instead of updating all 1.5 billion parameters, LoRA freezes the base model and trains small adapter matrices. QLoRA adds 4-bit quantization to the frozen weights. Result: training fits in 16GB VRAM with minimal quality loss."

2. **Unsloth:** "Optimized Triton kernels for the LoRA forward/backward pass. Without Unsloth, I was hitting OOM errors even with QLoRA. With it, training was stable and roughly 2x faster than standard HuggingFace."

**Model Selection (brief):**

*"I chose Qwen 2.5 Coder 1.5B Instruct. It's small enough to run locally on my M1 Mac, already specialized for code, and genuinely open source. Deepseek is technically better on benchmarks, but at 671B parameters it defeats the purpose of local deployment."*

**What I'd Change:**

*"Looking back, I'd spend more time on data quality and less on training optimization. The model learned exactly what I taught it—including the bugs in my dataset. No amount of clever training tricks fixes bad data."*

---

### Section 4: The Evaluation Pipeline

**The Problem (1 paragraph):**

*"Loss curves don't tell you if code runs. A model can achieve low perplexity while generating syntactically broken DAGs. I needed evaluation that matched how the code would actually be used."*

**The 3-Tier Pipeline:**

Present each tier with its purpose and implementation:

**Tier 1: Syntax Validation**
*"Standard Python AST parsing. If `ast.parse()` fails, the DAG is unusable. This catches missing colons, unbalanced parentheses, and invalid Python. It's the minimum bar."*

**Tier 2: Domain Validation**
*"A custom Airflow parser that checks Airflow-specific constraints: unique task IDs (duplicates cause silent overwrites), acyclic dependencies (cycles prevent DAG loading), and presence of actual DAG definitions (not just Python code). This catches code that's valid Python but broken Airflow."*

**Tier 3: Semantic Evaluation (LLM-as-Judge)**
*"Structural validity doesn't mean the code is good. I used Claude Sonnet to grade valid DAGs on three criteria:"*

- Idiomatic usage: Does it use the right operators, or wrap everything in PythonOperator?
- Hallucination check: Does it import non-existent modules or use fake parameters?
- Instruction adherence: Does it actually do what was asked?

*"The LLM judge revealed problems the parser couldn't catch—like the test utility imports that are valid Python but don't exist in production Airflow."*

**Cost Note:**

*"Running 1,000+ evaluations through Claude's Batch API cost about $X. Cheaper than I expected, and the qualitative feedback was invaluable for understanding failure modes."*

---

### Section 5: What I'd Do Differently

**Keep this concrete and actionable:**

1. **Filter training data aggressively:** "Exclude everything under `tests/`, `test_*.py`, and any file importing from non-public namespaces. The 15 minutes spent on filtering would have saved hours of debugging hallucinations."

2. **Audit synthetic data for homogeneity:** "Sample 50-100 generated instructions and look for repeated patterns before training. The '20' problem was obvious in hindsight but invisible when I was focused on volume."

3. **Start with smaller experiments:** "I trained on 10k samples immediately. I should have trained on 1k first, evaluated, identified issues, then scaled. The iteration loop matters more than the dataset size."

**Closing for Part 1:**

*"The fine-tuned model is better at writing Airflow code than the baseline—when it works. The failures taught me more than the successes. In Part 2, I'll cover deployment: getting this model running locally at 65 tokens/second on a Mac M1, and integrating it into actual development workflows."*

---

---

## Part 2: Deploying an SLM Locally—65 Tokens/Second on a Mac M1

**Core Promise:** Show the inference reality—why it's different from training, how to optimize for local hardware, and how to integrate the model into real workflows.

**Opening Hook:**

*"Training is about throughput—process as many samples as possible. Inference is about latency—generate tokens fast enough to be useful. These require completely different optimizations. A setup that's great for training can be unusable for inference."*

Brief context: *"I wanted to run my Airflow model locally. No API calls, no cloud costs, no data leaving my machine. The challenge: making a 1.5B model responsive enough to use interactively."*

---

### Section 1: The Inference Landscape

**Why Inference Is Different (2-3 paragraphs):**

Explain the fundamental constraint shift:

*"During training, you process batches in parallel—hundreds of sequences at once. GPU utilization is high because there's always work to do. During inference, you generate one token at a time, sequentially. Each token depends on all previous tokens. You can't parallelize the generation itself."*

*"This makes inference memory-bandwidth bound, not compute bound. The bottleneck isn't how fast you can multiply matrices—it's how fast you can load model weights from RAM to the processor. On my M1 Mac, the GPU can do the math faster than memory can feed it data."*

**The Stack Options (present as a journey, not a list):**

*"I tried three approaches:"*

1. **HuggingFace Transformers (CPU):** "4.19 tokens/second. I waited 10 seconds for 'Hello World.' Immediately ruled out for interactive use."

2. **HuggingFace Transformers (MPS/Metal):** "Better, but still under 20 t/s. The Python overhead and lack of inference-specific optimizations hurt."

3. **llama.cpp with Metal:** "64.52 tokens/second. 15x faster than the CPU baseline. This is what I shipped."

**Why llama.cpp Won:**

*"llama.cpp is C++ optimized specifically for inference. It uses GGUF model format (quantized weights in a single file), supports Metal acceleration on Mac, and implements KV-cache to avoid recomputing attention for previous tokens. The Python wrapper (`llama-cpp-python`) gives you the speed without leaving the Python ecosystem."*

---

### Section 2: Quantization—Did It Break the Code?

**The Concern:**

*"I quantized the model to Q4 (4-bit weights) to fit in memory and improve throughput. The obvious question: does aggressive quantization break code generation? Syntax is unforgiving—one wrong character and the code doesn't run."*

**The Test:**

Describe your validation approach:

*"I ran the same evaluation pipeline from Part 1 on the quantized model. Syntax validity, domain checks, LLM grading—the full suite."*

**The Results:**

*"Surprisingly, Q4 quantization didn't significantly degrade code quality. Syntax error rates were within 1-2% of the full-precision model. The model still made the same semantic mistakes (test utility hallucinations, instruction following issues), but quantization didn't add new failure modes."*

*"This matches what others have found: code generation is more robust to quantization than tasks requiring precise numerical reasoning. The model needs to get tokens right, not floating-point values."*

**Show a concrete example:**

Include a side-by-side of the same prompt with FP16 vs Q4 output, demonstrating equivalent quality.

---

### Section 3: Benchmark Numbers (Your Actual Hardware)

**Be specific about the setup:**

*"All benchmarks on Mac Pro M1 with 16GB unified memory, running macOS Sonoma. Model: Qwen 2.5 Coder 1.5B Instruct, Q4_K_M quantization, 2048 context length."*

**The Numbers:**

| Configuration | Tokens/Second | Notes |
|--------------|---------------|-------|
| HuggingFace CPU | 4.19 | Unusable for interactive work |
| HuggingFace MPS | ~18 | Acceptable but sluggish |
| llama.cpp Metal | 64.52 | Primary configuration |
| llama.cpp batch=8 | 184 | Throughput mode for bulk generation |

**Interpretation:**

*"At 65 t/s, the model generates a typical DAG (200-400 tokens) in 3-6 seconds. That's fast enough to be useful in a development workflow—comparable to waiting for a linter or test suite."*

*"The 184 t/s throughput mode uses parallel decoding for batch inference. Useful for generating test sets or evaluation samples, not for interactive use."*

---

### Section 4: Integration—CLI and MCP Server

**The Goal:**

*"A model that only runs in a Jupyter notebook isn't a tool. I wanted two integration points: a CLI for quick generation and an MCP server for IDE integration."*

**CLI Design:**

Describe the actual interface:

```bash
# Generate a DAG from a prompt
airflownet generate "Create a DAG that extracts data from S3, transforms with pandas, and loads to Snowflake"

# Interactive chat mode
airflownet chat
```

*"The CLI handles server lifecycle automatically. If the llama.cpp server isn't running, it spawns one in the background. No manual setup required."*

**MCP Server (for Cursor/Claude integration):**

*"MCP (Model Context Protocol) lets you expose tools to AI-powered IDEs. I wrapped the generation endpoint as an MCP tool, so I can ask Cursor to 'generate an Airflow DAG for this task' and it calls my local model."*

Show the actual integration:

*"The server runs on `localhost:8000` with an OpenAI-compatible API. Any tool that speaks to OpenAI can point at the local server instead."*

**Architecture Diagram:**

Include a simple diagram showing:
```
User → CLI/MCP → Server Manager → llama.cpp Server → Model (GGUF)
```

---

### Section 5: What I Learned About Local Inference

**Memory Bandwidth Is Everything:**

*"On consumer hardware, you're almost always memory-bound. The M1's 200GB/s memory bandwidth determines throughput more than its GPU compute capability. This is why quantization helps so much—smaller weights mean fewer bytes to move."*

**Quantization Is Free (For Code):**

*"I expected Q4 quantization to hurt quality. It didn't. For code generation, the precision loss is negligible. This might not hold for tasks requiring numerical reasoning, but for syntax-heavy generation, aggressive quantization is a free performance win."*

**The Python Wrapper Is Fine:**

*"I initially tried running the raw llama.cpp C++ server, expecting the Python wrapper to add latency. The overhead was negligible—within 5% of raw C++ performance. Use whatever's easier to integrate."*

---

### Section 6: What's Next

**Keep this honest about limitations and future work:**

1. **Data quality:** "The hallucination problem from Part 1 persists regardless of inference optimization. Better data filtering would improve results more than any inference trick."

2. **Larger models:** "1.5B is the limit for comfortable M1 inference. With a beefier Mac (M2 Ultra, 64GB+) or dedicated GPU, 7B models become viable. The same optimization stack applies."

3. **Speculative decoding:** "For even faster inference, speculative decoding uses a smaller draft model to propose tokens that the main model verifies. I haven't implemented this yet, but it's the logical next step for latency reduction."

**Closing:**

*"The goal was a model that runs locally, generates valid Airflow DAGs, and responds fast enough to be useful. At 65 t/s on a Mac M1, that goal is met—with caveats. The model is better than baseline at idiomatic Airflow code but still hallucinates when it hits edge cases in the training data."*

*"The tools are accessible. Colab for training, llama.cpp for inference, a few hundred lines of Python for integration. The hard part isn't the infrastructure—it's the data. Get that right and the rest follows."*

---

---

## Appendix: Style Reminders for Writing

When converting this outline to prose:

1. **Don't bold everything.** Use bold sparingly for key terms on first introduction.

2. **Write in paragraphs.** The tables and lists in this outline are for your reference. The actual post should flow as prose, with lists only where they genuinely help (like the benchmark comparison).

3. **Cut the meta-commentary.** Don't write "In this section, we'll explore..." Just start exploring.

4. **Keep code examples realistic.** When showing DAG code, use real operator names and plausible task structures.

5. **Admit uncertainty.** "I suspect..." and "I haven't tested..." are fine. Don't claim comprehensive knowledge you don't have.

6. **End sections without fanfare.** You don't need a summary sentence after every section. Sometimes you can just move to the next topic.

7. **The CTAs should be specific.** Instead of "What do you think about fine-tuning?" try "Has anyone else seen the synthetic data homogeneity problem? I'm curious if there's a standard deduplication approach."