#!/bin/bash

cd /workspace/plot/ckpt_perf
./run_all_ckpt.sh
./run_all_elastic.sh
python3 proc_data.py

cd /workspace/plot/overall_results
python3 read_data.py --ckpt_perf /workspace/ckpt_perf/ckpt_perf.json --results_dir /workspace/results
