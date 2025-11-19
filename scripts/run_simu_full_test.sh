#!/bin/bash

cd /workspace/simulator
mkdir -p logs_pai
env BEFORE_SCALE_LIMIT=5 TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 415 --seed_end 415 --task_num_begin 200 --task_num_end 200 --log_dir logs_pai
env BEFORE_SCALE_LIMIT=5 TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 100 --seed_end 100 --task_num_begin 240 --task_num_end 240 --log_dir logs_pai
env BEFORE_SCALE_LIMIT=5 TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 370 --seed_end 370 --task_num_begin 200 --task_num_end 200 --log_dir logs_pai
python3 examples/extract_log_runtime.py --log_dir logs_pai

mkdir -p logs_ms
env BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 300 --seed_end 300 --task_num_begin 170 --task_num_end 170 --log_dir logs_ms
env BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 852 --seed_end 852 --task_num_begin 210 --task_num_end 210 --log_dir logs_ms
env BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 300 --seed_end 300 --task_num_begin 150 --task_num_end 150 --log_dir logs_ms
python3 examples/extract_log_runtime.py --log_dir logs_ms