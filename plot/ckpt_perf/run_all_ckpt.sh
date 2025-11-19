#!/bin/bash

mkdir -p ckpt_perf_res/

python3 -u ckpt_perf.py --config-file /workspace/geth/examples/gat_ogbn-arxiv.py 2>&1 | grep "\[" | tee ckpt_perf_res/gat.txt
python3 -u ckpt_perf.py --config-file /workspace/geth/examples/gcn_ogbn-arxiv.py 2>&1 | grep "\[" | tee ckpt_perf_res/gcn.txt
python3 -u ckpt_perf.py --config-file /workspace/geth/examples/gin_ogbn-arxiv.py 2>&1 | grep "\[" | tee ckpt_perf_res/gin.txt
python3 -u ckpt_perf.py --config-file /workspace/geth/examples/sage_ogbn-arxiv.py 2>&1 | grep "\[" | tee ckpt_perf_res/sage.txt
