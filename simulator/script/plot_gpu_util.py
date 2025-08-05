import re
import sys
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import os
from scipy.signal import savgol_filter
import glob

def parse_log_file(log_file):
    """解析日志文件，提取GPU利用率数据
    
    Args:
        log_file (str): 日志文件路径
        
    Returns:
        tuple: (timestamps, machine_data)
            - timestamps: 时间戳列表
            - machine_data: 字典，键为机器名，值为利用率百分比列表
    """
    time_pattern = re.compile(r'.*\[(\d+)\.\d+\]: cluster_info=')
    machine_data = defaultdict(dict)
    total_data = {} 
    
    with open(log_file, 'r') as f:
        for line in f:
            # 匹配时间戳
            time_match = time_pattern.match(line)
            if not time_match:
                continue
                
            timestamp = int(time_match.group(1))
            
            # 提取每台机器的数据
            machines = re.findall(r"'machine_\d+': \{.*?\}", line)
            total_allocated = 0
            total_gpus = 0
            for machine in machines:
                name = re.search(r"'machine_\d+'", machine).group(0).strip("'")
                allocated = int(re.search(r"'allocated_gpus': (\d+)", machine).group(1))
                total = int(re.search(r"'total_gpus': (\d+)", machine).group(1))
                
                utilization = (allocated / total) * 100 if total > 0 else 0
                machine_data[name].update({timestamp: utilization})
                
                total_allocated += allocated
                total_gpus += total
                
            total_utilization = (total_allocated / total_gpus) * 100
            total_data[timestamp] = total_utilization
    
    return machine_data, total_data

def compress_data(util_dict):
    compressed_util_dict = dict()
    prev_time = None
    prev_util = None
    
    for time, util in util_dict.items():
        if prev_util != util:
            if prev_util is not None:
                compressed_util_dict.update({prev_time: prev_util})
            compressed_util_dict.update({time: util})
        prev_time = time
        prev_util = util
    compressed_util_dict.update({prev_time: prev_util}) # final 可能会重复，不过没有关系
    
    return compressed_util_dict

def plot_total_utilization(total_data, output_file='total_gpu_utilization.svg'):
    """单独绘制总GPU占用率图
    
    Args:
        total_data (dict): 总占用率数据 {timestamp: utilization}
        output_file (str): 输出图片路径
    """
    plt.figure(figsize=(10, 5))
    times = sorted(total_data.keys())
    utils = [total_data[time] for time in times]
    
    plt.plot(times, utils, linestyle='-', color='red', linewidth=2, label='Total GPU Utilization')
    plt.title('Total GPU Utilization Over Time')
    plt.xlabel('Time')
    plt.ylabel('Utilization (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    print(f"Total utilization plot saved as {output_file}")
    plt.close()

def plot_gpu_utilization(machine_data, output_file='gpu_utilization.svg'):
    """绘制GPU利用率图表（支持动态每行子图数量）
    
    Args:
        timestamps (list): 时间戳列表
        machine_data (dict): 机器数据字典
        cols_per_row (int): 每行显示的子图数量，默认为2
        output_file (str): 输出图片路径
    """
    machines = sorted(machine_data.keys())
    num_machines = len(machines)
    if num_machines == 1:
        cols_per_row = 1
    else:
        cols_per_row = 2
    rows = math.ceil(num_machines / cols_per_row)  # 计算需要的行数
    
    # 创建子图网格
    fig, axes = plt.subplots(
        rows, 
        cols_per_row, 
        figsize=(5 * cols_per_row, 4 * rows),  # 动态调整图像大小
        squeeze=False  # 确保axes总是二维数组
    )
    fig.suptitle(f'GPU Utilization Over Time ({cols_per_row} per row)', fontsize=16)
    
    # 绘制每个机器的数据
    for idx, machine in enumerate(machines):
        util_dict = machine_data[machine]
        times = sorted(util_dict.keys())
        utils = [util_dict[time] for time in times]
        
        print(f"Plotting {machine} with {len(times)} data points")
        row = idx // cols_per_row
        col = idx % cols_per_row
        ax = axes[row, col]
        
        
        ax.plot(times, utils, linestyle='-', marker='', markersize=3, label=machine)
        ax.set_title(machine)
        ax.set_xlabel('Time')
        ax.set_ylabel('Utilization (%)')
        ax.grid(True)
    
    # 隐藏多余的子图
    for idx in range(num_machines, rows * cols_per_row):
        row = idx // cols_per_row
        col = idx % cols_per_row
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.close()  # 防止在非交互环境下重复显示

def plot_multi_total_utilization(data_dict, output_file='multi_total_gpu_utilization.svg'):
    plt.figure(figsize=(10, 5))
    # times = sorted(set().union(*[d.keys() for d in data_dict.values()]))
    for type in sorted(data_dict.keys()):
        data = data_dict[type]
        # utils = [data.get(x, None) for x in times]
        width = 2 if "elastic" in type else 1.5
        alpha = 0.8 if "elastic" in type else 0.6
        plt.plot(data.keys(), data.values(), linestyle='-', alpha=alpha, linewidth=width, label=type)
    
    plt.title('Total GPU Utilization Over Time')
    plt.xlabel('Time')
    plt.ylabel('Utilization (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    print(f"Total utilization plot saved as {output_file}")
    plt.close()

def smooth_data(data_dict, type: str="savgol"):
    if type == "savgol":
        times = list(data_dict.keys())
        values = list(data_dict.values())
        values[-1] = 0
        window_size = 5  # 窗口大小（必须是奇数）
        poly_order = 2   # 多项式阶数
        smoothed_values = savgol_filter(values, window_size, poly_order)
        smoothed_dict = {time: value for time, value in zip(times, smoothed_values)}
    elif type == "sample":
        step = 40
        times = list(data_dict.keys())
        values = list(data_dict.values())
        smoothed_times = times[::step]
        smoothed_values = values[::step]
        smoothed_values[-1] = 0
        smoothed_dict = {time: value for time, value in zip(smoothed_times, smoothed_values)}
    else:
        step = 40
        times = list(data_dict.keys())[::step]
        values = list(data_dict.values())[::step]
        window_size = 7  # 窗口大小（必须是奇数）
        poly_order = 3   # 多项式阶数
        smoothed_values = [100 if value > 100 else 0 if value < 0 else value for value in savgol_filter(values, window_size, poly_order)]
        smoothed_values[-1] = 0
        smoothed_dict = {time: value for time, value in zip(times, smoothed_values)}
    return smoothed_dict

def get_data(log_file):
    gpu_file = os.path.splitext(os.path.basename(log_file))[0] + "_gpu_util" + '.svg'
    total_file = os.path.splitext(os.path.basename(log_file))[0] + "_total_util" + '.svg'
    smooth_total_file = os.path.splitext(os.path.basename(log_file))[0] + "_smooth_total_util" + '.svg'
    machine_data, total_data = parse_log_file(log_file)
    smooth_total_data = smooth_data(total_data, "other")
    compressed_machine = {m: compress_data(d) for m, d in machine_data.items()}
    compressed_total = compress_data(total_data)
    compressed_smooth_total = compress_data(smooth_total_data)
    plot_gpu_utilization(compressed_machine, gpu_file)
    plot_total_utilization(compressed_total, total_file)
    plot_total_utilization(compressed_smooth_total, smooth_total_file)

def parse_filename(filename):
    # 匹配 {type}_simulation_seed_{seed}_task_{task_num}
    pattern = r"(.+)_simulation_seed_(\d+)_task_(\d+)"
    match = re.match(pattern, filename.stem)  # filename.stem 去掉后缀
    if match:
        return {
            "type": match.group(1),
            "seed": int(match.group(2)),
            "task_num": int(match.group(3)),
            "filepath": filename
        }
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='Plot GPU utilization from log file.')
    parser.add_argument('--log_dir', type=str, help='Path to the log dir containing GPU utilization data log.', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to the output dir.', default=None)
    args = parser.parse_args()

    skip_type = ["elastic_pre"]

    from pathlib import Path

    log_dir = Path(args.log_dir)
    log_files = list(log_dir.glob("*.log"))
    parsed_logs = []
    for file in log_files:
        parsed = parse_filename(file)
        if parsed:
            parsed_logs.append(parsed)
    
    if args.output_dir is None:
        output_dir = log_dir / "figure"
    else:
        output_dir = args.output_dir
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame(parsed_logs)
    grouped = df.groupby(["seed", "task_num"])

    for (seed, task_num), group in grouped:
        plt.figure(figsize=(10, 6))
        plt.title(f"Seed={seed}, Task={task_num}")
        all_data = {}
        for _, row in group.iterrows():
            type = row["type"]
            if type in skip_type:
                continue
            log_file = row["filepath"]
            machine_data, total_data = parse_log_file(log_file)
            smooth_total_data = smooth_data(total_data, "other")
            # compressed_machine = {m: compress_data(d) for m, d in machine_data.items()}
            # compressed_total = compress_data(total_data)
            compressed_smooth_total = compress_data(smooth_total_data)
            all_data[type] = compressed_smooth_total
        plot_multi_total_utilization(all_data, output_dir / f"seed_{seed}_task_{task_num}.svg")

if __name__ == "__main__":
    main()
