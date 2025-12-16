Done
- dag miner
- instruction creator using Claude with multithreading 


To do
- fine tuning: 
- - improve llm evaluation: to improve I need to visualise how current dag files are evaluated. To do so, I want to have a final dataset with, model type, sys prompt, instruction (airlfow or generic code), original output, model output and evaluation.
- - beautify: 1. add tokens as secrets to inject in a colab notebook, add README and explanation of how those models have been build
- mem bound improvements:
- - on M1: try MLX, quantised KV cache, not spec decoding (I am already almost at mem bandwidht, having 8 small models is bandwidth consuming, it is beneficial only if draft is really good, but we are talking about very small models here...)
-  deploy GPUs inference: increase workers + vLLM pagedAttention and run a benchmark (TTFT, TPS, cost efficency=) + dedicated GPU quantisazton (AWQ)
- inference repo beautify. Has to be runnable stand alone and I want to write a blog with that (how do I calculate CTX, batch, input output?)
- quantisation, distillation, pruning?


Next steps
- fine tuning improvements:
- - scale single GPU fine tuning with LoRa, Flash-attn2, bigger context and more data (add new skills) -> I can also use bigger model. ATM it take 30/40 min
- - evaluate different quant and models
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