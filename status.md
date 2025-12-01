Done
- dag miner
- instruction creator using Claude with multithreading 


To do
- time to generate dags with llama server optimised params
- fine tuning
- mem bound improvements:
- - on M1: try MLX, quantised KV cache, spec decoding (not sure since 8 workers)
-  - change GPU: vLLM pagedAttention (works with CUDA), bigger machine and more workers
- inference repo beautify. Has to be runnable stand alone and I want to write a blog with that (how do I calculate CTX, batch, input output?)


Next steps
- LLM as teacher that generate new type of instructions (debugging)
- generalise api calls to any LLM 
- add a way to score code complexity (euristic) and analyse performances for different complexity and version
- inject known failures (based on my experience and on mosto common I find in source code) into current correct DAGs implementations dataset for creating a troubleshooting dataset for finetuning


Dag miner overview
- it mines github official repo since it has lots of example for all the versions
- check python file correctness with ast and compile
- out of scope airflow compiler or unit test(too complicated, it has to be worth it)
- it (only) checks if there are internal import with simle euristics
- it adds metadata for analyse SML performance later on


Open points