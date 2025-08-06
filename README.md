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

## GETH 端到端性能实验

使用 GETH 在模拟器上的八机八卡集群上随机生成任务，并给出不同调度方案下的模拟运行结果。

我们使用 pai.csv 或者 ms.csv 作为任务轨迹文件，二者分别来自于和阿里集群和微软 Azure 的公开数据集。
可以通过修改 `/workspace/simulator/examples/random_trace_generator.py` 中的 `trace_csv_path`
为 `/workspace/simulator/pai.csv` 或 `/workspace/simulator/ms.csv` 来选择不同的任务轨迹文件。

### PAI 数据集实验

```bash
cd /workspace/simulator
env BEFORE_SCALE_LIMIT=5 python3 examples/run_all_parallel.py --seed_begin 415 --seed_end 415 --task_num_begin 200 --task_num_end 200
env BEFORE_SCALE_LIMIT=5 python3 examples/run_all_parallel.py --seed_begin 100 --seed_end 100 --task_num_begin 240 --task_num_end 240
python3 examples/extract_log_runtime.py
```

### MS 数据集实验

```bash
cd /workspace/simulator
env BEFORE_SCALE_LIMIT=5 python3 examples/run_all_parallel.py --seed_begin 300 --seed_end 300 --task_num_begin 170 --task_num_end 170
env BEFORE_SCALE_LIMIT=5 python3 examples/run_all_parallel.py --seed_begin 852 --seed_end 852 --task_num_begin 210 --task_num_end 210
python3 examples/extract_log_runtime.py
```

## 模拟器性能测试实验

使用模拟器在单机八卡上随机生成任务，并给出不同调度方案下的模拟运行结果。
结果保存在 `/workspace/simulator/logs/` 目录下：

```bash
cd /workspace/simulator
python3 examples/run_all_parallel.py --seed_begin 26 --seed_end 26 --phy
python3 examples/extract_log_runtime.py
```

将模拟器的调度方案转化为实际运行的任务格式：

```bash
cd /workspace/simulator
# elastic_pre_sjf
python3 script/parse_log.py logs/elastic_pre_sjf_simulation_seed_26_task_20.log --output logs/elastic_pre_sjf.json
# elastic_pre
python3 script/parse_log.py logs/elastic_pre_simulation_seed_26_task_20.log --output logs/elastic_pre.json
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
script results/elastic_pre_sjf.log
env GETH_ENABLE_PRESCALE=1 python3 tests/st/train.py /workspace/simulator/logs/elastic_pre_sjf.json
# elastic_pre
script results/elastic_pre.log
env GETH_ENABLE_PRESCALE=1 python3 tests/st/train.py /workspace/simulator/logs/elastic_pre.json
# sjf
script results/sjf.log
python3 tests/st/train.py /workspace/simulator/logs/sjf.json
# fifo
script results/fifo.log
python3 tests/st/train.py /workspace/simulator/logs/fifo.json
```

处理结果并绘图：

```bash
cd /workspace/geth
# elastic_pre_sjf
python3 script/ansi_log_processor.py results/elastic_pre_sjf.log
python3 script/plot_timeline_from_real.py results/elastic_pre_sjf.log --output results/elastic_pre_sjf_timeline.png
# elastic_pre
python3 script/ansi_log_processor.py results/elastic_pre.log
python3 script/plot_timeline_from_real.py results/elastic_pre.log --output results/elastic_pre_timeline.png
# sjf
python3 script/ansi_log_processor.py results/sjf.log
python3 script/plot_timeline_from_real.py results/sjf.log --output results/sjf_timeline.png
# fifo
python3 script/ansi_log_processor.py results/fifo.log
python3 script/plot_timeline_from_real.py results/fifo.log --output results/fifo_timeline.png
```
