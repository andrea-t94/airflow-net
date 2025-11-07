Done
- dag miner
- instruction creator using Claude with multithreading 


To do
- super slow (for retrials?) + hitting 429 (too many requests) + I pay a lot -> try with asyncio + rate limiting and cheaper model/smaller prompt (one instruction), less retry
- add version in instruction
- create an mcp that enable semantic search on the whole airflow github, based on version



Next steps
- airflow dag compiler to increase data quality
- LLM as teacher that generate more instructions per dag and even new dags
- test asyncio for LLM call, might be more performant
- inject known failures (based on my experience and on mosto common I find in source code) into current correct DAGs implementations dataset for creating a troubleshooting dataset for finetuning


Dag miner overview
- it mines github official repo since it has lots of example for all the versions
- check python file correctness with ast and compile
- out of scope airflow compiler or unit test(too complicated, it has to be worth it)
- it (only) checks if there are internal import with simle euristics
- it adds metadata for analyse SML performance later on


Open points