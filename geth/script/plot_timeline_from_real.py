import re
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json

def parse_task_simu_info(log_lines):
    # 正则表达式匹配 job 信息
    pattern = r"DEBUG\s+\|\s+__main__:run_task:60\s+\-\s+\[\s+\d+\.\d+\]:\s+(\{.*?\})"
    log_content = ''.join(log_lines)
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    tasks = {}
    for match in matches:
        try:
            # 将字符串转换为字典
            simu_info = json.loads(match.replace("'", '"'))
            tasks[simu_info['job_id']] = simu_info
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            continue
    
    return tasks

def parse_task_real_info(log_lines):
    real_data = defaultdict(dict)
    
    # 匹配任务创建日志（提取时间、任务名、任务文件）
    created_pattern = re.compile(
        r'\[(\d+\.\d+)\].*Created task: Task\['
        r'task_name=(\w+),\s*task_file=([^,]+),'
    )
    
    # 匹配任务回复开始训练日志
    recoveried_pattern = re.compile(
        r'\[(\d+\.\d+)\].*Task\['
        r'task_name=(\w+),.*All agent recoveried'
    )
    
    get_scale_resources = re.compile(
        r'\[(\d+\.\d+)\].*Task\['
        r'task_name=(\w+),.*agents_num=(\d+),.*All agents get resources for scale'
    )
    
    # 匹配任务完成日志
    finished_pattern = re.compile(
        r'\[(\d+\.\d+)\].*Task\['
        r'task_name=(\w+),.*status=TaskStatus\.RUNNING.*Finished'
    )
    
    
    for line in log_lines:
            
        # 处理任务创建日志
        created_match = created_pattern.search(line)
        if created_match:
            time, task_name, task_file = created_match.groups()
            task_type = os.path.splitext(os.path.basename(task_file))[0]
            real_data[task_name].update({
                'create_time': float(time),
                'task_type': task_type,
                'task_file': task_file
            })
            
        # 处理任务回复开始训练日志
        recoveried_match = recoveried_pattern.search(line)
        if recoveried_match:
            time, task_name = recoveried_match.groups()
            real_data[task_name]['assigned_gpus_finish_time'] = float(time)
            
        get_scale_resources_match = get_scale_resources.search(line)
        if get_scale_resources_match:
            time, task_name, scale_num = get_scale_resources_match.groups()
            if 'scales' not in real_data[task_name]:
                real_data[task_name]['scales'] = []
            real_data[task_name]['scales'].append([float(time), int(scale_num)])
        
        # 处理任务完成日志
        finished_match = finished_pattern.search(line)
        if finished_match:
            time, task_name = finished_match.groups()
            real_data[task_name]['finish_time'] = float(time)
    
    return real_data

def plot_task_timeline(real_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 颜色映射
    unique_types = {data['task_type'] for data in real_data.values()}
    colors = plt.cm.tab20.colors[:len(unique_types)]
    color_map = {t: colors[i] for i, t in enumerate(unique_types)}
    
    # 计算最大时间值用于动态偏移
    max_time = max(data['finish_time'] for data in real_data.values())
    x_margin = max_time * 0.15
    
    for i, (task_name, data) in enumerate(real_data.items()):
        duration = data['assigned_gpus_finish_time'] - data['submit_time']
        ax.barh(i, duration,
                left=data['submit_time'],
                color=color_map[data['task_type']],
                alpha=0.5,
                edgecolor='black')
        duration = data['finish_time'] - data['assigned_gpus_finish_time']
        ax.barh(i, duration,
                left=data['assigned_gpus_finish_time'],
                color=color_map[data['task_type']],
                alpha=0.7,
                edgecolor='black')
        
        label = f"{task_name} ({data['task_type']})"
        label_x = data['finish_time'] + (max_time * 0.02)  # 初始偏移2%
        text_obj = ax.text(label_x, i, label,
                            va='center', ha='left',
                            fontsize=9)
        
        renderer = fig.canvas.get_renderer()
        bbox = text_obj.get_window_extent(renderer=renderer)
        bbox = bbox.transformed(ax.transData.inverted())
        
        if bbox.x1 > max_time + x_margin:
            x_margin = (bbox.x1 - max_time)
    
    # 设置x轴范围（自动扩展右侧空间）
    ax.set_xlim(left=0, right=max_time + x_margin)
    
    # 自定义图表样式
    ax.set_yticks(range(len(real_data)))
    ax.set_yticklabels([f"{name}" for name in real_data.keys()])
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_title('Real Task Execution Timeline', fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    # 添加图例
    # handles = [plt.Rectangle((0,0),1,1, color=color_map[t]) for t in unique_types]
    # ax.legend(handles, unique_types, title='Task Types', 
    #           loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize task execution timeline from log file.')
    parser.add_argument('log_file', help='Path to the log file')
    parser.add_argument('--name', default='ElasGNN')
    parser.add_argument('--pre_scale', action='store_true')
    parser.add_argument('--output', default='timeline_real.png', 
                       help='Output image path (default: timeline_real.png)')
    args = parser.parse_args()

    try:
        with open(args.log_file) as f:
            log_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{args.log_file}' not found")
        return
    
    simu_info = parse_task_simu_info(log_lines)
    real_data = parse_task_real_info(log_lines)
    
    real_data = dict(sorted(real_data.items(), key=lambda item: item[1]['create_time']))
    
    # if args.pre_scale:
    #     real_data = dict(sorted(real_data.items(), key=lambda item: (item[1]['assigned_gpus_finish_time'], item[1]['finish_time'])))
    # else:
    #     real_data = dict(sorted(real_data.items(), key=lambda item: (item[1]['create_time'], item[1]['finish_time'])))
    
    simu_jcts = []
    real_jcts = []
    for task_name in real_data.keys():
        simu = simu_info[task_name]
        real = real_data[task_name]
        if 'finish_time' not in real:
            print(f"Warning: Task {task_name} has no finish_time in real data, skipping.")
            continue
        real['submit_time'] = simu['submit_time']
        simu_jct = simu['finish_time'] - simu['submit_time']
        real_jct = real['finish_time'] - simu['submit_time']
        simu_jcts.append(simu_jct)
        real_jcts.append(real_jct)
    print("\nSummary:")
    print(f"Name: {args.name}")
    
    average_simu_jct = sum(simu_jcts) / len(simu_jcts)
    average_real_jct = sum(real_jcts) / len(real_jcts)
    average_jct_error = abs((average_real_jct - average_simu_jct) / average_simu_jct)
    
    print(f"Average Simulated JCT: {average_simu_jct:.2f}s")
    print(f"Average Real JCT:      {average_real_jct:.2f}s")
    print(f"Average JCT Error:     {average_jct_error:.2%}")
    
    simu_end_time = max(simu['finish_time'] for simu in simu_info.values())
    phy_end_time = max(real['finish_time'] for real in real_data.values())
    print(f"Simulated Make Span:   {simu_end_time:.2f}s")
    print(f"Real Make Span:        {phy_end_time:.2f}s")
    print(f"Make Span Error:       {(phy_end_time - simu_end_time) / phy_end_time:.2%}")
    
    # fig = plot_task_timeline(real_data)
    # fig.savefig(args.output, dpi=300, bbox_inches='tight')
    # print(f"Timeline saved to {args.output}")

if __name__ == "__main__":
    main()