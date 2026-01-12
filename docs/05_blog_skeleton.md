# Blog Skeleton
The blog will be divided into 2 main parts:
1. I explain the overarching objective and fine tune a model and evaluate it
2. I deploy it locally

## 0. Intro
The overarching objective is to share my learning journey on how to apply LLM/SLM to a real world use case: I'll finetune and deploy an SLM locally (explain my local stack as well). 

Why?
1. I want to learn how to apply SLM to a real world use case and I want to share my learnings from a pov of a practictioner coming from classical ML (e.g. differences against classical ML, stack/algorithms for training and inference LLM/SLM)
2. I believe that in the future SLM will be the standard for many tasks because they can be deployed locally with your personal stack, allowing to build secure applications and spend less money on 3rd party LLMs or infra to deploy an LLM
3. Literature demonstrates that SLM can outperform or perform in par with LLMs in a number of tasks, but it is still a nascent field and there is no much literature on how to apply it to real world use cases

What?
I will use build an SLM that can write airflow dags (called AirflowNet). As said at this stage I don't care about the results, I want to show the (iterative) process that I use to create and deploy a model.
I will talk about some improvements that I plan to do in case I see interest from the community.

How?
I will blog my learnings as said and I'll open source my code + data.
The code contains both the research code and a working CLI and MCP server that I've built to understand how inference work for LLMs and what are some of the most common ways to deploy them locally.

## 1. Modelling
The idea is to showcase that the process of finetuning an SLM locally is similar to standard model development process (from defining the objective to evaluate it, as defined in some books like ml system design).  
I want to show that the process is not that different from standard model development, apart from the specificities of SLMs (each ML model, not only LLM, has its own specificities).

### 1.1 Objective
I want to finetune an SLM to write airflow dags (called AirflowNet).  
I will basically do LLM knowledge distillation: I will use an LLM to teach the SLM how to write airflow dags, under the assumption that the latest frontier LLMs are really good at writing and evaluating airflow dags (since I am use them for this task).  - add references about that -  

### 1.2 Data Collection
Since coding is a difficult task, that requires a perfect sintax knowledge, I'll start the dataset with official examples, per different airflow version, directly taken from airflow official github.

### 1.3 Data Preprocessing
The dataset, in chatML format, has been created asking an LLM (in this case I've created a client for Claude API) to generate 3 instructions per dag file collected. The instructions are generated in a way that they are different from each other and from the dag file, but they are still related to it. The augmentation is useful to have more data, since SLM/LLM are data hungry. 
I'll also clean the file a bit, removing unnecessary comments, otherwise the SLM will learn it as well. 

### 1.4 Model Selection
I have opted for using Qwen coder 2.5 1.5B Instruct, since it is open source, specialised in coding (which, as said, is more complex that simple writing) and it can fit into my local stack. 

### 1.5 Model Training
I'll dive on how training works for LLM and why it is so expensive. This will help me to justify the choices I make in the following sections (it might also be a unique paragraph without the sub-sections).
#### 1.5.1 The Compute Stack: Google Colab with GPU A100
#### 1.5.2 Using Unsloth for 2x faster training (since it rewrites some custom kernels)
#### 1.5.3 Technique: QLoRA (Low-Rank Adaptation with Quantization)
#### 1.5.4 Artifacts: Saving Adapters + Merging to GGUF

### 1.6 Model Evaluation
I'll show that the model has learned to write airflow dags with its specific sintax. I'll also show some of the pitfalls of the current approach.

### 1.7 Potential next steps
I'll propose solutions to overcome them, highlighting the iterative process of model development.
#### 1.7.1 Fixing Catastrophic Forgetting: Mixing more generic Python data to restore syntax
#### 1.7.2 Reducing Hallucinations: Better filtering of niche libraries vs "Teacher" LLM generation
#### 1.7.3 Robustness: Injecting failures to train "Troubleshooting" capability



## 2. Deployment
Here the objective is instead to highight the engineering side, which instead is pretty much difference. In fact, wheter locally or in cloud, the inference process is different compared to the training phase, which is something new on the ML landscape. The fact that there are specific trick to speed up inference (e.g. KV cache, paged attention) and that there are specific tools to deploy the model (e.g. vLLM, LLM Cache) are something new on the ML landscape.

### 2.1 Objective

### 2.2 Model Inference Engine (The Foundation)
#### 2.2.1 Why inference is different (KV Cache, PagedAttention)
#### 2.2.2 The Local Stack: Deep dive on llama.cpp (GGUF, Quantization internals)
#### 2.2.3 Benchmarks: Native (llama.cpp) vs Python (HF+bitsandbytes)
* Metric 1: Tokens/sec (The obvious one)
* Metric 2: RAM usage (Critical for local)
* Metric 3: Model Quality (Did Q4 quantization break the code syntax?)
* Analysis: Compute vs Memory Bound bottlenecks (Why Mac is bandwidth starved)

### 2.3 Deployment Architecture 1: The "Unix" Way (CLI)
#### 2.3.1 CLI implementation details
#### 2.3.2 Use Case: Piping and scripting

### 2.4 Deployment Architecture 2: The "Agentic" Way (MCP Server)
#### 2.4.1 What is MCP and why it matters for IDEs (Cursor/Windsurf)
#### 2.4.2 Server implementation details

### 2.5 Potential next steps
#### 2.5.1 Pushing Local Limits: Deeper Quantization vs MLX (Apple Silicon native)
#### 2.5.2 Performance Analysis: Finding the theoretical max (Compute/Mem bound)
#### 2.5.3 Scaling Up: From local CLI to concurrent serving (vLLM)
