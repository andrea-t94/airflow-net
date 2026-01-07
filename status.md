Done
- dag miner
- instruction creator using Claude with multithreading 


To do
- deploy sth locally
- - fine tune on QWEN CODER!! ANd make BASE MODEL NAME as config var clearly visible
- - update docs research findings and changelog, they will be used for the blog
- write two blog:
     1. modelling (objective is local to mimic llm, find data, fine tune and eval) + next steps
     2. deploying (system designed, how llama.cpp inference works and local performances evaluation) + next steps


Next steps
- dataset improvements: (see changelog 22)
- - scrape more DAG files
- - use better LLM to create instructions
- - add context to the instructions (e.g. what tech stack is used)
- - end-to-end evaluation of DAG files
- - new type of data

- local deployment improvements:
- - on M1: try MLX
- - in general: quantised KV cache, not spec decoding (it is beneficial only if draft is really good, but we are talking about very small models here...)
- - more quantisation, distillation, pruning?

- deployment improvements:
- - probably make sense to have this as a separate side project were I migrate from local to multiple GPUs
- - increase workers + vLLM pagedAttention and run a benchmark (TTFT, TPS, cost efficency=)
- - dedicated GPU quantisazton (AWQ) 
- - LMCache


- fine tuning improvements:
- - scale single GPU fine tuning with LoRa, Flash-attn2, bigger context and more data (add new skills) -> I can also use bigger model. ATM it take 30/40 min
- - evaluate different quant and models
- LLM as teacher that generate new type of instructions (debugging)
- generalise api calls to any LLM 
- add a way to score code complexity (euristic) and analyse performances for different complexity and version
- inject known failures (based on my experience and on mosto common I find in source code) into current correct DAGs implementations dataset for creating a troubleshooting dataset for finetuning

data imrpovements:
- more data on Dags and generic code
- better quality and preprocessing
- excllude internal libraries
- add more airflow versions

model improvements:
- e.g. performance with different code compelxity -> might need better model
- add better dag file parser that can become a tool