Done
- dag miner
- instruction creator using Claude with multithreading 


To do
- make possible to run benchmarsk with higher concurrent workload. It was failing for OOM issues
- try vLLM instead?
- finalise inference fine tuning, save all the scripts and findings in a separate folder, it will become a blogpost
- time to generate dags with qwne: one-time eval of the existed raw and processed, to check and eventually exclude the bad dags instructions with new DAG parser + use best inference framework to geenrate dags


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