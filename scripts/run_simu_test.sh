#!/bin/bash

cd /workspace/simulator
env TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 26 --seed_end 26 --phy --log_dir logs_simu
python3 examples/extract_log_runtime.py --log_dir logs_simu

python3 script/parse_log.py logs_simu/elastic_pre_sjf_simulation_seed_26_task_20.log --output logs_simu/elastic_pre_sjf.json
python3 script/parse_log.py logs_simu/sjf_simulation_seed_26_task_20.log --output logs_simu/sjf.json
python3 script/parse_log.py logs_simu/fifo_simulation_seed_26_task_20.log --output logs_simu/fifo.json