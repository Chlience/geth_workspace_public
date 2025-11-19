# GETH

## 环境配置

安装模拟器依赖：

```bash
pip install simpy
```

配置运行时环境：

```bash
su wpb
source /workspace/scripts/env_vars.sh
cd /workspace/gccl/build && make install
cd /workspace/ragdoll && ./build.sh
cd /workspace/geth && pip install -e .
```

## 模拟器性能测试实验

使用模拟器在单机八卡上随机生成任务，并给出不同调度方案下的模拟运行结果：

```bash
/workspace/scripts/run_simu_test.sh
```

实际运行任务：

```bash
cd /workspace/geth
mkdir results

# elastic_pre_sjf
script results/elastic_pre_sjf_phy.log
env GETH_ENABLE_PRESCALE=1 python3 tests/st/train.py /workspace/simulator/logs_simu/elastic_pre_sjf.json
# Ctrl+D exit script

# sjf
script results/sjf_phy.log
python3 tests/st/train.py /workspace/simulator/logs_simu/sjf.json
# Ctrl+D exit script

# fifo
script results/fifo_phy.log
python3 tests/st/train.py /workspace/simulator/logs_simu/fifo.json
# Ctrl+D exit script
```

输出结果

```bash
/workspace/scripts/get_simu_result.sh
```

## GPU 使用率实验

```bash
/workspace/scripts/run_gpu_alloc_test.sh
```

结果在 `/workspace/simulator/logs_gpu_alloc/` 目录下。

## 模拟器大实验

### PAI 数据集实验

```bash
/workspace/scripts/run_simu_full_test.sh
```

结果在 `/workspace/simulator/logs_pai/` 和 `/workspace/simulator/logs_ms/` 目录下。

## Scale 性能对比实验

```bash
/workspace/scripts/run_scale_test.sh
```

结果在 `/workspace/plot/overall_results/overall_results.json` 文件中