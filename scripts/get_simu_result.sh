#!/bin/bash

cd /workspace/geth
python3 script/ansi_log_processor.py results/elasgnn_phy.log
python3 script/ansi_log_processor.py results/sjf_phy.log
python3 script/ansi_log_processor.py results/yarn-cs_phy.log

python3 script/plot_timeline_from_real.py results/elasgnn_phy_processed.log --name ElasGNN
python3 script/plot_timeline_from_real.py results/sjf_phy_processed.log --name SJF
python3 script/plot_timeline_from_real.py results/yarn-cs_phy_processed.log --name YARN-CS
