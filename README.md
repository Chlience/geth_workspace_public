# GETH

## 环境配置

安装模拟器依赖：

```bash
pip install simpy
```

配置运行时环境：

```bash
cd /workspace/gccl/build && make install
cd /workspace/ragdoll && ./build.sh
cd /workspace/geth && pip install -e .
```

## 模拟器性能测试实验

使用模拟器在单机八卡上随机生成任务，并给出不同调度方案下的模拟运行结果。
结果保存在 `/workspace/simulator/logs/` 目录下：

```bash
cd /workspace/simulator
env TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 26 --seed_end 26 --phy
python3 examples/extract_log_runtime.py
```

将模拟器的调度方案转化为实际运行的任务格式：

```bash
cd /workspace/simulator
# elastic_pre_sjf
python3 script/parse_log.py logs/elastic_pre_sjf_simulation_seed_26_task_20.log --output logs/elastic_pre_sjf.json
# sjf
python3 script/parse_log.py logs/sjf_simulation_seed_26_task_20.log --output logs/sjf.json
# fifo
python3 script/parse_log.py logs/fifo_simulation_seed_26_task_20.log --output logs/fifo.json
```

实际运行任务：

```bash
cd /workspace/geth
mkdir results

# elastic_pre_sjf
script results/elastic_pre_sjf_phy.log
env GETH_ENABLE_PRESCALE=1 python3 tests/st/train.py /workspace/simulator/logs/elastic_pre_sjf.json
# Ctrl+D exit script

# sjf
script results/sjf_phy.log
python3 tests/st/train.py /workspace/simulator/logs/sjf.json
# Ctrl+D exit script

# fifo
script results/fifo_phy.log
python3 tests/st/train.py /workspace/simulator/logs/fifo.json
# Ctrl+D exit script
```

处理结果并绘图：

```bash
cd /workspace/geth
# elastic_pre_sjf
python3 script/ansi_log_processor.py results/elastic_pre_sjf_phy.log
# sjf
python3 script/ansi_log_processor.py results/sjf_phy.log
# fifo
python3 script/ansi_log_processor.py results/fifo_phy.log
```

## GPU 使用率实验

```bash
env ENABLE_CLUSTER_INFO=1 ENABLE_TASK_INFO=1 BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 852 --seed_end 852 --task_num_begin 210 --task_num_end 210
```

## 模拟器大实验

### PAI 数据集实验

```bash
cd /workspace/simulator
mkdir -p logs_pai
env BEFORE_SCALE_LIMIT=5 TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 415 --seed_end 415 --task_num_begin 200 --task_num_end 200 --log_dir logs_pai
env BEFORE_SCALE_LIMIT=5 TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 100 --seed_end 100 --task_num_begin 240 --task_num_end 240 --log_dir logs_pai
env BEFORE_SCALE_LIMIT=5 TRACE=PAI python3 examples/run_all_parallel.py --seed_begin 370 --seed_end 370 --task_num_begin 200 --task_num_end 200 --log_dir logs_pai
python3 examples/extract_log_runtime.py --log_dir logs_pai
```

### MS 数据集实验

```bash
cd /workspace/simulator
mkdir -p logs_ms
env BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 300 --seed_end 300 --task_num_begin 170 --task_num_end 170 --log_dir logs_ms
env BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 852 --seed_end 852 --task_num_begin 210 --task_num_end 210 --log_dir logs_ms
env BEFORE_SCALE_LIMIT=5 TRACE=MS python3 examples/run_all_parallel.py --seed_begin 300 --seed_end 300 --task_num_begin 150 --task_num_end 150 --log_dir logs_ms
python3 examples/extract_log_runtime.py --log_dir logs_ms
```