# GETH

## 环境配置

配置运行时环境：

```bash
su wpb
source /workspace/scripts/env_vars.sh
cd /workspace/gccl/build && make install
cd /workspace/ragdoll && ./build.sh
cd /workspace/geth && pip install -e .
pip install simpy strip_ansi
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

script results/elasgnn_phy.log
env GETH_ENABLE_PRESCALE=1 python3 tests/st/train.py /workspace/simulator/logs_simu/elasgnn.json
# Ctrl+D exit script

script results/sjf_phy.log
python3 tests/st/train.py /workspace/simulator/logs_simu/sjf.json
# Ctrl+D exit script

script results/yarn-cs_phy.log
python3 tests/st/train.py /workspace/simulator/logs_simu/yarn-cs.json
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

```bash
/workspace/scripts/run_simu_full_test.sh
```

结果在 `/workspace/simulator/logs_pai/` 和 `/workspace/simulator/logs_ms/` 目录下。

## Scale 性能对比实验

```bash
/workspace/scripts/run_scale_test.sh
```

结果在 `/workspace/plot/overall_results/overall_results.json` 文件中
