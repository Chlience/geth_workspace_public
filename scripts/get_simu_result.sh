#!/bin/bash

cd /workspace/geth
python3 script/ansi_log_processor.py results/elastic_pre_sjf_phy.log
python3 script/ansi_log_processor.py results/sjf_phy.log
python3 script/ansi_log_processor.py results/fifo_phy.log

python3 script/plot_timeline_from_real.py results/elastic_pre_sjf_phy_processed.log --name ElasGNN
python3 script/plot_timeline_from_real.py results/sjf_phy_processed.log --name SJF
python3 script/plot_timeline_from_real.py results/fifo_phy_processed.log --name YARN-CS