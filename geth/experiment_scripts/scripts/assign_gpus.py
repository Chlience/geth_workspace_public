import os
import argparse
from itertools import combinations

def generate_test_commands(model, dataset, wait_time=None):
    commands = []
    results_dir = f"results/{model}_{dataset}/"
    commands.append(f"mkdir -p {results_dir}")
    
    NUMA_0 = [0, 1, 2, 3]
    NUMA_1 = [4, 5, 6, 7]
    
    for size in range(1, 9):
        for l_size in range(max(size - 4, 1), min(size, 4) + 1):
            r_size = size - l_size
            gpus_list = NUMA_0[:l_size] + NUMA_1[:r_size]
            wait_arg = f"--wait_time {wait_time} " if wait_time else ""
            cmd = f"python3 -u tests/st/training_no_scale_cmdline_special.py {model}_{dataset} --initial_gpus '{gpus_list}' {wait_arg}2>&1 | tee {results_dir}no_scale_lr_{l_size}_{r_size}.txt"
            commands.append(cmd)
    
    return commands

def main():
    parser = argparse.ArgumentParser(description='Generate test commands.')
    parser.add_argument('--output', type=str, default='run_all_tests.sh',
                      help='Output script filename (default: run_all_tests.sh)')
    args = parser.parse_args()
    
    models = [
        "gat",
        "gcn",
        "gin",
        "sage",
    ]
    datasets = [
        "reddit",
        "yelp",
        "ogbn-products",
        "ogbn-arxiv",
    ]
    wait_times = {
        "reddit": {
            "default": 40,
        },
        "yelp": {
            "default": 40,
        },
        "ogbn-products": {
            "default": 50,
        },
        "ogbn-arxiv": {
            "default": 30,
        }
    }
    
    all_commands = []
    
    # 为每个数据集生成命令
    for model in models:
        all_commands.append(f"# {model}")
        all_commands.append("")
        for dataset in datasets:
            wait_time = wait_times[dataset].get(model, wait_times[dataset].get("default", None))
            all_commands.extend(generate_test_commands(model, dataset, wait_time=wait_time))
            all_commands.append("")
    
    # 将命令写入脚本文件
    with open(args.output, "w") as f:
        f.write("\n".join(all_commands))

if __name__ == "__main__":
    main()