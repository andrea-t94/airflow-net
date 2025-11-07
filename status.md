Done
- dag miner
- instruction creator using Claude with multithreading 


To do
- super flow (for retrials?) + hitting 429 (too many requests) + I pay a lot -> try with asyncio + rate limiting and cheaper model/smaller prompt (one instruction), less retry
- scrape for qa
- analyse most common Dag failures from source code
- inject failures into current correct DAGs for debugging


Next steps
- airflow dag compiler to increase data quality
- LLM as teacher that generate more instructions per dag and even new dags
- test asyncio for LLM call, might be more performant


Dag miner overview
- it mines github official repo since it has lots of example for all the versions
- check python file correctness with ast and compile
- out of scope airflow compiler or unit test(too complicated, it has to be worth it)
- it (only) checks if there are internal import with simle euristics
- it adds metadata for analyse SML performance later on


Open points