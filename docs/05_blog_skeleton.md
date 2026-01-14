# Blog Skeleton

## MetaData / Strategy: From Toy to Tool

**Blog Series Title**: *From Toy to Tool: The Engineering Reality of Local SLMs*

**The Focus**:
*   This is **not** a tutorial on "how to run a model".
*   This **is** a deep dive into the **lifecycle of specialized Small Language Models (SLMs)**.
*   We treat the SLM not as a magical black box, but as a engineered software component: finetuned for a specific job (Airflow DSL), deployed with specific constraints (Local/Privacy), and integrated into specific workflows (Unix Pipes & Agentic IDEs).

**The Target Audience**:
*   **The Disenchanted Practitioner**: ML Engineers and Data Pros who are tired of generic "Chat with your PDF" demos and want to see rigorous application of SLMs.
*   **The Local-First Developer**: Engineers who value privacy, zero-latency, and offline capability, and want to know if local SLMs are actually viable for code generation.
*   **The Agent Builder**: People looking to build complex systems where low-cost, high-speed local models act as "function routers" or "drafters" for larger cloud models.

---
The blog will be divided into 3 main parts:
1. I explain how am I and why I am cool and the overarching objective
2. I fine tune a model and evaluate it
3. I deploy it locally


## 0. Intro
Who am I?
I am a practitioner coming from classical ML (e.g. differences against classical ML, stack/algorithms for training and inference LLM/SLM). - SHARE SMALL PARTS OF MY JOURNEY USING MY RESUME, -  I've started as analyst from mgmgt engineering and I am now working as a data/ML engineer in one of the biggest eu scaleup, Flix. I've always had big passion for AI since when it was called DL (with some sporadic projects I've worked in).
I have started not knowing much about ML and data, but I have been learning myself over the years and grew till senior.  

Why I am doing this?
Overarching objective: dive into AI (LLM) and share my pov as a data practictioner coming from "classical" ML
I think as an experienced data/ML practitioner, knowing more about latest advancement in AI is essential and also fun.
This is not the first time I am learning something, but this time I want to share my learning journey.
This first serie of blog posts is about how to apply LLM/SLM to a real world use case: I'll finetune and deploy an SLM locally (explain my local stack as well). 
Once I've learned enough and I'm familiar the main concepts/methods/tools from an application point of view, I'll move to more advanced and internal topics to increase my understanding and imrpvoe application performances(e.g. build an LLM from scratch, build a attention algorithm from scratch, optimise training/inference by writing custom kernels etc.).
This is sort of opposite direction of how you learn at university, and more similar to how I learn at work (learnig by doing and when you need).

Why you should read me?
This is definitely not a begineer friendly blog series, it is/will be meant for people who already have a solid background in ML/data engineering and want to learn more about latest AI advancements in a pragmatic and end-to-end way.
Most of the content I find online is really theoretical and/or covers a small part of the process, but I want to show you the full picture, from gathering the data till the inner workings of the models and methods around AI, so that we can learn how to apply them at best.



#### The first series of blog posts will be on applying LLM/SLM to a real world use case.
Why you should read this?
I want you to feel less intimidated by applying LLMs to your use cases.
1. I believe that in the future SLM will be the standard for many tasks because they can be deployed locally with your personal stack, allowing to build secure applications and spend less money on 3rd party LLMs or infra to deploy an LLM. Literature demonstrates that SLM can outperform or perform in par with LLMs in a number of tasks, but it is still a nascent field and there is no much literature on how to apply it to real world use cases end to end (from data collection to deployment).
2. I want to show you that, most of the process is extremely similar to the classical ML process you, as a data practictioner, already knows.

What?
I will finetune an SLM that can write airflow dags (called AirflowNet). As said, at this stage I don't care about the results, I want to show the (iterative) process that I use to create and deploy a model. 
I plan to show you from a data practictioner coming from the classical background (what does classical mean?). I will show you that, most of the process is extremely similar to the classical ML process. 
I will talk about some improvements that I plan to do in case I see interest from the community.

How?
I will blog my learnings as said and I'll open source my code + data. The blog will be shared in 2 parts, one per week.
The code contains both the research code and a working CLI and MCP server that I've built to understand how inference work for LLMs and what are some of the most common ways to deploy them locally.
I'd be really happy if you will share some feedbacks, questions or suggestions for next series.


## 1. Part 1: Fine tune the SLM
The idea is to showcase that the process of finetuning an SLM locally is similar to standard model development process: from gathering the data to evaluate it, as defined in some books like ml system design.  
I want to show that the process is not that different from standard model development and, if you are an experienced data practitioner, you should find it familiar.

### 1.1 Objective
I want to finetune an SLM to write airflow dags (called AirflowNet). I will use an instruct SLM, since my goal is to fine tune a model good at following instructions.  -  SMALL INTRO ON AIRFLOW - 
They are ideal for applications like code generation, summarization, Q&A, and automation, where predictable and structured results are needed, unlike base models which are more generalized or chat models focused on conversation. Instruct models excel at understanding intent and delivering direct answers.

I will basically do LLM knowledge distillation: I will use an LLM to help me prep the training dataset (instructions) for the SLM and to evaluate the generated dags.
Main assumptions/ideas:
1. latest frontier LLMs are really good at writing and evaluating airflow dags, therefore my goal is to make my SLM be as good as them at airflow dag file generation  - ADD REFERENCES ABOUT THAT, distill labs -  
2. SLM are proved to perform on par or even better on specific task vs LLMs, that's why I want to show you that it is possible to build a SLM that can write airflow dags


### 1.2 Data Collection
Coding is a difficult task: it requires perfect sintax knowledge of the language, knowledge of the language idiom (ie., airflow) and the ability to follow instructions to generate something working and useful.
To make sure the SLM is learning to write airflow dags, I'll start the dataset with official examples, per different airflow version, directly taken from airflow official github so that I am sure I am following the airflow way of doing things. 
Obviously this can be extended to unofficial examples, but with the risk of learning bad practices and would require lot of process to clean the data. 
To make the model (which I'll talk in the next session) remember also python syntax, I'll also use magpie dataset, which is a dataset of extracted code from Qwen 2.5 Coder (our base model) - ADD REFERENCES ON MAGPIE - 
Same goes for using the LLM to generate dag files: high risk of model hallucination plus the cost of using an LLM to generate the dataset.

### 1.3 Data Preprocessing
The airflow dataset has been created asking an LLM (in this case I've created a client for Claude API) to generate 3 instructions per dag file collected. The instructions are generated in a way that they are slightly different from each other for the same dag file. The augmentation is useful to have more data, since SLM/LLM are data hungry. 
From 04_project_learning.md: I've decided to generate **3 instructions per request**. This proved to be really cost effective using batch messages (since I don't have latency requirements), I spent <$2 (with one request for one instruction at time >$5).
I've also cleaned up the dag files, since they were containing lot of comments upfront which I didn't want the model to learn. I didn't instead remove or add internal libraries/imports for time reasons, hoping the model will learn how to cope with that (we'll see later on that is not the case).
I have made sure to have enough data to train the model, and kept also python related instructions (again from magpie, no instruction generation needed) to avoid catastrophic forgetting. 
Finally, I've applied the ChatML format, with which Qwen coder 2.5 Instruct is compatible - SHORT PRIMER ON TEMPLATE FORMATS -  

### 1.4 Model Selection
I have opted for using Qwen coder 2.5 1.5B Instruct, since it is open source, specialised in coding (which, as said, is more complex that simple writing) and it can fit into my local stack. 
This blog helped me deciding for qwen: https://huggingface.co/blog/daya-shankar/open-source-llms
It's one of the best open source LLM family for coding, it's fully open source and they provide a distilled version that fits into my mac.
Eventhoudh there were some better model in the leaderboard like Deepseek (https://livebench.ai/#/?Reasoning=a&Coding=a&Mathematics=a&Data+Analysis=a&Language=a&IF=a&sort=Coding+Average&openweight=true), I've opted for Qwen 2.5 coder since it was distributing a 1.5B distilled version already in huggingface (https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)

### 1.5 Model Fine tuning
I'll dive on how training works for LLM and why it is so expensive. This will help me to justify the choices I make in the following sections (it might also be a unique paragraph without the sub-sections).
#### 1.5.1 A primer on (plain) model training and why it is so expensive
A small intro on how training works for LLM, highlighting that is expensive (quadratic memory and runtime cost). This is to explain why usage of GPUs is so important.
Since I don't have my own GPUs, I rely on colab, but feel free to use your own hardware. Since I do have Pro, I opted for A100 to dramatically speed up fine tunning vs free T4 GPUs (explain why it's faster).
Explain how long it will take only using T4 or A100 GPUs
#### 1.5.2 Fine tuning optimisation with Unsloth and QLoRA
Moreover, to speed up the process, I've used Unsloth, which is a library with optimised triton kernels (is in my roadmap to write a blog on it). The second optimisation component is by using PEFT method called LoRA (Low-Rank Adaptation) combined with quantization (QLoRA). Also this is something I'm not going to dive too much. - ADD REFERENCES ON UNLOSTH, QLORA, PEFT - 
#### 1.5.3 Fine tuning process
I just show the overall script, without going into details on the code. I'll just highlight if I did anything particular, besides the aformentioned optimisations. It's just plain unsloth script.

### 1.6 Model Evaluation
I've used a custom ast and airflow dag parser (CTA to help me and link of the code) + using LLM as a judge - LINK TO LLM EVALs blog recommended - The LLM as a judge prompt as been fine tuned based on the ground truth data (that should be considered good).
The model has been evaluated on a set of airflow dags, highlighting that it has learned to write airflow dags with its specific sintax. Compared to the baseline model, it has shown to have learned specific airflow syntax, like operators.
I'll also show some of the pitfalls of the current approach:
1. catastrophic forgetting on python syntax -> add python data
2. hallucination on niche libraries -> filter them out or increase sample size
3. overfitting on internal libraries (e.g. test libraries) -> filter them out
4. creating more complex dags, and therefore is more prone to errors

### 1.7 Conclusion and next steps (whit CTA)
I'll wrap up the potential improvements:
- the ones highilighted on the section above
- increase model and dataset size since the models are better with scale
- with the last option, I can try to improve finetuning engineering (e.g. training on multiple GPUs, stronger or dynamic quantisation etc.).
Asking also the crowd feedback on what I could improve or what they'd like to see.




## 2. Deployment
Here the objective is instead to highight the engineering effort on the inference side.
In fact, wheter locally or in cloud, the inference process is different compared to the training phase, which is something new on the ML landscape. The fact that there are specific trick to speed up inference (e.g. KV cache, paged attention etc.) and that there are specific tools to deploy the model (e.g. vLLM, LLM Cache) are something new on the ML landscape.
In this blog I'll dive a bit on how to deploy an LLM locally, building a chatbot CLI and an MCP server utilisable via e.g. Claude code.
I'll also share the learnings on how I optimised for local inference and which tool I've used.

### 2.1 Objective
I want to share how inference works and why it needs ad hoc solution. I also want to show what are the most common tools and techniques to deploy an LLM locally. 
Finally I want to show how I optimised my local inference and what are the 2 ways to deploy it (as a "choatbot" via CLI or as a tool via MCP server).

### 2.2 Model Inference Engine
#### 2.2.1 A primer on inference and why it is different from training
#### 2.2.2 What are the most common techniques to improve inference performance (GGUF, Quantization internals, KVCache, FlashAttn)
#### 2.2.3 Why I opted for Llama.cpp
Intro on what is llama.cpp and why it is so fast. Then I show the benchmarks and analysis.
* Metric 1: Tokens/sec (The obvious one)
* Metric 2: RAM usage (Critical for local)
* Metric 3: Model Quality (Did Q4 quantization break the code syntax?)
* Analysis: Compute vs Memory Bound bottlenecks (Why Mac is bandwidth starved)
#### 2.2.4 My inference engine
I explain why simply running the server isn't enough - we need a "Brain" to control it.
*   **The Abstraction Layer**: Why I built `engine.py` instead of raw API calls everywhere.
    *   *Separation of Concerns*: Decoupling the LLM backend from the application logic (CLI/MCP).
*   **Implementation Details**:
    *   **Universal Client**: I use the standard `openai` Python SDK to connect to `llama.cpp`.
        *   *Benefit*: It makes the backend swappable (e.g., to GPT-4 or vLLM) without code changes.
    *   **Prompt Engineering as Code**: Encapsulating the "Airflow Expert" persona.
        *   Injecting System Prompts and handling ChatML formatting transparently.
    *   **Robust Output Parsing**: Solving the "Chatty Model" problem.
        *   Implementation of `_extract_code` to strip markdown fences and comments, ensuring clean executable Python code.
    *   **Deterministic Configuration**: Enforcing `temperature=0.1` for reliable code generation.

### 2.3 Deployment Architecture 1: The CLI (Human-to-Model Interface)
*   **The Concept**: Treating SLMs as "Unix Utilities".
    *   Just like `grep` or `awk`, a specialized SLM should be a fast, local tool that does one thing well (in this case, writing Airflow code).
*   **Why specific to SLMs?**:
    *   *Zero Latency & Privacy*: No network calls means you can pipe sensitive DAG logic directly into it.
    *   *Offline Capable*: It becomes a dependable tool in your dev environment, not a service you rent.
*   **The Architecture**: A thin client wrapper that abstracts the complexity of model loading (GGUF) and prompt formatting, exposing a simple text-in/text-out interface.

### 2.4 Deployment Architecture 2: The MCP Server (Agent-to-Model Interface)
*   **The Concept**: The "Microservice" for AI Agents.
    *   Instead of a human chatting with the model, we expose the SLM as a "Tool" for other, smarter models (like Claude in Cursor or Windsurf).
*   **Why this changes the game for SLMs**:
    *   *Specialization*: Your general IDE Agent (Claude) doesn't need to know every Airflow edge case. It can delegate that to your fine-tuned expert SLM.
    *   *Cost Efficiency*: You offload the heavy token generation (writing verbose DAGs) to a free local model, keeping the paid API model as the high-level orchestrator.
*   **The Architecture**: Implementing the Model Context Protocol (MCP) to standardize how the local model "advertises" its capabilities to the IDE.

### 2.5 Conclusion and next steps (with CTA)
I'll wrap up the learning and the next steps
    1. to improve existing solution (stronger quantisation, using MLX)
    2. to explore completely new techniques/tools (e.g. I'd like to scale this with concurrent requests, so I'll try vLLM). 

