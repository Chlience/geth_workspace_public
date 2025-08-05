import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
import os, argparse

default_gpu_num = [1, 2, 4, 8]

def parse_task_log(log_file):
    pattern = re.compile(r'\[(\d+)\.\d+\]: task_info=\{(.*?)\}$')
    gpu_data = defaultdict(dict)

    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue

            timestamp = int(match.group(1))
            tasks_str = match.group(2)

            # 提取所有任务的min_gpu
            tasks = re.findall(r"'job_\d+': \{.*?'min_gpu': (\d+).*?\}", tasks_str)

            # 统计当前时间点各min_gpu的任务数
            current_counts = {int(min_gpu): 0 for min_gpu in default_gpu_num}
            
            for min_gpu in tasks:
                current_counts[int(min_gpu)] += 1

            # 记录数据
            for min_gpu, count in current_counts.items():
                gpu_data[min_gpu].update({timestamp: count})

    return gpu_data

def compress_data(task_data):
    compressed_task_data = defaultdict(dict)
    
    # 首先压缩时间戳和每个机器的数据
    for min_gpu, count_dict in task_data.items():
        compressed_count_dict = dict()
        prev_time = None
        prev_count = None
        
        for time, count in count_dict.items():
            if prev_count != count:
                if prev_count is not None:
                    compressed_count_dict.update({prev_time: prev_count})
                compressed_count_dict.update({time: count})
            prev_time = time
            prev_count = count
        compressed_count_dict.update({prev_time: prev_count}) # final 可能会重复，不过没有关系
        
        compressed_task_data[min_gpu] = compressed_count_dict
    
    return compressed_task_data

def plot_task_counts(gpu_data, output_file='queue_task_count.svg'):
    """绘制各min_gpu的活跃任务数随时间变化图（秒级精度）
    
    Args:
        gpu_data (dict): parse_task_log返回的数据
        output_file (str): 输出图片路径
    """
    plt.figure(figsize=(14, 7))
    
    # 为每个min_gpu绘制曲线
    for min_gpu, count_dict in sorted(gpu_data.items()):
        times = sorted(count_dict.keys())
        counts = [count_dict[time] for time in times]
        plt.plot(times, counts, linestyle='-', marker='', markersize=3, label=f'min_gpu={min_gpu}')

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Queue Task Count', fontsize=12)
    plt.title('Queue Tasks by min_gpu Requirement (Second-level Precision)', fontsize=14)
    
    # 优化时间轴显示
    ax = plt.gca()
    ax.grid(True)
    ax.legend(fontsize=10)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot GPU utilization from log file.')
    parser.add_argument('--log_file', type=str, help='Path to the log file containing GPU utilization data.', required=True)
    parser.add_argument('--output', type=str, help='Output file name for the plot.')
    args = parser.parse_args()
    log_file = args.log_file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.splitext(os.path.basename(log_file))[0] + "_queue_task_count" + '.svg'
    
    task_data = parse_task_log(log_file)
    compressed_task_data = compress_data(task_data)
    original_size = sum(len(v) for v in task_data.values())
    compressed_size = sum(len(v) for v in compressed_task_data.values())
    print(f"Data compressed from {original_size} points to {compressed_size} points "
          f"({compressed_size/original_size*100:.1f}% of original)")
    plot_task_counts(compressed_task_data, output_file)