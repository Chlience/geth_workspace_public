import os
import argparse

def generate_test_commands(model, dataset, wait_time=None):
    commands = []
    results_dir = f"results/{model}_{dataset}/"
    commands.append(f"mkdir -p {results_dir}")
    
    # scale
    scale_size_pairs = []
    for i in range(1, 9):
        for j in range(1, 9):
            if i != j:
                scale_size_pairs.append((i, j))
    
    for start, end in scale_size_pairs:
        wait_arg = f"--wait_time {wait_time} " if wait_time else ""
        cmd = f"python3 -u tests/st/training_scale_cmdline.py {model}_{dataset} --start_size {start} --end_size {end} {wait_arg}2>&1 | tee {results_dir}scale_{start}_{end}.txt"
        commands.append(cmd)
        
    # no_scale
    
    no_scale_size = range(1, 9) 
    for size in no_scale_size:
        wait_arg = f"--wait_time {wait_time} " if wait_time else ""
        cmd = f"python3 -u tests/st/training_no_scale_cmdline.py {model}_{dataset} --size {size} {wait_arg}2>&1 | tee {results_dir}no_scale_{size}.txt"
        commands.append(cmd)
        
    # no_scale_no_micro
    
    no_scale_no_micro_size = range(1, 9)
    arxiv_special = False
    arviv_jump = True
    for size in no_scale_no_micro_size:
        wait_arg = f"--wait_time {wait_time + 20} " if wait_time else ""
        cmd = f"python3 -u tests/st/training_no_scale_cmdline.py {model}_{dataset} --size {size} {wait_arg}2>&1 | tee {results_dir}no_scale_no_micro_{size}.txt"
        if (size == 5 or size == 7) and dataset == "ogbn-arxiv":
            if arviv_jump:
                continue
            if arxiv_special:
                cmd = "GCCL_PART_OPT=\"RECUR_MINI\" " + cmd
        else:
            cmd = "GCCL_PART_OPT=\"METIS\" " + cmd
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
            "gat":30,
            "gcn":30,
            "gin":25,
            "sage":25,
        },
        "yelp": {
            "default": 20,
        },
        "ogbn-products": {
            "default": 40,
            "gat":30,
            "gcn":30,
            "gin":25,
            "sage":25,
        },
        "ogbn-arxiv": {
            "default": 8,
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