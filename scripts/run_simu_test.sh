#!/bin/bash

cd /workspace/simulator
mkdir -p logs_simu

env TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 26 --seed_end 26 --phy --log_dir logs_simu
mv logs_simu/elastic_pre_sjf_simulation_seed_26_task_20.log logs_simu/elasgnn.log
mv logs_simu/sjf_simulation_seed_26_task_20.log logs_simu/sjf.log
mv logs_simu/fifo_simulation_seed_26_task_20.log logs_simu/yarn-cs.log

python3 examples/extract_log_runtime.py --log_dir logs_simu

python3 script/parse_log.py logs_simu/elasgnn.log --output logs_simu/elasgnn.json
python3 script/parse_log.py logs_simu/sjf.log --output logs_simu/sjf.json
python3 script/parse_log.py logs_simu/yarn-cs.log --output logs_simu/yarn-cs.json
