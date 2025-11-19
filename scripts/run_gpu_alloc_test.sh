#!/bin/bash

cd /workspace/simulator
env ENABLE_CLUSTER_INFO=1 ENABLE_TASK_INFO=1 BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 852 --seed_end 852 --task_num_begin 210 --task_num_end 210 --log_dir logs_gpu_alloc
