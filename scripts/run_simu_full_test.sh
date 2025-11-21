#!/bin/bash

cd /workspace/simulator
mkdir -p logs_pai
env BEFORE_SCALE_LIMIT=5 TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 415 --seed_end 415 --task_num_begin 200 --task_num_end 200 --log_dir logs_pai
env BEFORE_SCALE_LIMIT=5 TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 100 --seed_end 100 --task_num_begin 240 --task_num_end 240 --log_dir logs_pai
env BEFORE_SCALE_LIMIT=5 TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 370 --seed_end 370 --task_num_begin 200 --task_num_end 200 --log_dir logs_pai

mv logs_pai/elastic_pre_sjf_simulation_seed_415_task_200.log logs_pai/elasgnn_simulation_0.log
mv logs_pai/sjf_simulation_seed_415_task_200.log logs_pai/sjf_simulation_0.log
mv logs_pai/fifo_simulation_seed_415_task_200.log logs_pai/yarn-cs_simulation_0.log
mv logs_pai/elastic_pre_sjf_simulation_seed_100_task_240.log logs_pai/elasgnn_simulation_1.log
mv logs_pai/sjf_simulation_seed_100_task_240.log logs_pai/sjf_simulation_1.log
mv logs_pai/fifo_simulation_seed_100_task_240.log logs_pai/yarn-cs_simulation_1.log
mv logs_pai/elastic_pre_sjf_simulation_seed_370_task_200.log logs_pai/elasgnn_simulation_2.log
mv logs_pai/sjf_simulation_seed_370_task_200.log logs_pai/sjf_simulation_2.log
mv logs_pai/fifo_simulation_seed_370_task_200.log logs_pai/yarn-cs_simulation_2.log

python3 examples/extract_log_runtime.py --log_dir logs_pai

mkdir -p logs_ms
env BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 300 --seed_end 300 --task_num_begin 170 --task_num_end 170 --log_dir logs_ms
env BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 852 --seed_end 852 --task_num_begin 210 --task_num_end 210 --log_dir logs_ms
env BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 300 --seed_end 300 --task_num_begin 150 --task_num_end 150 --log_dir logs_ms

mv logs_ms/elastic_pre_sjf_simulation_seed_300_task_170.log logs_ms/elasgnn_simulation_0.log
mv logs_ms/sjf_simulation_seed_300_task_170.log logs_ms/sjf_simulation_0.log
mv logs_ms/fifo_simulation_seed_300_task_170.log logs_ms/yarn-cs_simulation_0.log
mv logs_ms/elastic_pre_sjf_simulation_seed_852_task_210.log logs_ms/elasgnn_simulation_1.log
mv logs_ms/sjf_simulation_seed_852_task_210.log logs_ms/sjf_simulation_1.log
mv logs_ms/fifo_simulation_seed_852_task_210.log logs_ms/yarn-cs_simulation_1.log
mv logs_ms/elastic_pre_sjf_simulation_seed_300_task_150.log logs_ms/elasgnn_simulation_2.log
mv logs_ms/sjf_simulation_seed_300_task_150.log logs_ms/sjf_simulation_2.log
mv logs_ms/fifo_simulation_seed_300_task_150.log logs_ms/yarn-cs_simulation_2.log

python3 examples/extract_log_runtime.py --log_dir logs_ms