import os
import re
import json
import argparse

parser = argparse.ArgumentParser(description='Extract runtime and speedup from simulation logs.')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory containing the log files.')
args = parser.parse_args()
log_dir = args.log_dir

output = {}

pattern = re.compile(r'(elastic_pre_sjf_|elastic_pre_|sjf_|fifo_|)?simulation_seed_(\d+)\.log')
pattern_with_task = re.compile(r'(elastic_pre_sjf_|elastic_pre_|sjf_|fifo_|)?simulation_seed_(\d+)_task_(\d+)\.log')
runtime_pattern = re.compile(r'Total Runtime: ([\d.]+) seconds')
avg_pattern = re.compile(r'Avg Completion Time: ([\d.]+) seconds')

with_task = False
for fname in os.listdir(log_dir):
    m = pattern_with_task.fullmatch(fname)
    if m:
        print("Processing task log:", fname)
        with_task = True
        prefix = m.group(1) or ''
        seed = m.group(2)
        task_num = m.group(3)
    else:
        m = pattern.fullmatch(fname)
        if m:
            if with_task:
                print(f"Can't deal with both task and non-task logs in the same directory: {fname}")
                continue
            print("Processing task log:", fname)
            prefix = m.group(1) or ''
            seed = m.group(2)
        else:
            continue
    if prefix == 'elastic_pre_sjf_':
        log_type = 'elastic_pre_sjf'
    elif prefix == 'elastic_pre_':
        log_type = 'elastic_pre'
    elif prefix == 'sjf_':
        log_type = 'sjf'
    else:
        log_type = 'fifo'
    path = os.path.join(log_dir, fname)
    with open(path, 'r') as f:
        lines = f.readlines()
    total_runtime = None
    avg_completion_time = None
    for line in lines:
        if total_runtime is None:
            m1 = runtime_pattern.search(line)
            if m1:
                total_runtime = float(m1.group(1))
        if avg_completion_time is None:
            m2 = avg_pattern.search(line)
            if m2:
                avg_completion_time = float(m2.group(1))
        if total_runtime is not None and avg_completion_time is not None:
            break
    if seed not in output:
        output[seed] = {}
    if with_task:
        if task_num not in output[seed]:
            output[seed][task_num] = {}
        output[seed][task_num][log_type] = {
            'total_runtime': total_runtime,
            'avg_completion_time': avg_completion_time
        }
    else:
        output[seed][log_type] = {
            'total_runtime': total_runtime,
            'avg_completion_time': avg_completion_time
        }

with open(os.path.join(log_dir, 'runtime_summary.json'), 'w') as f:
    json.dump(output, f, indent=2)

# 计算speedup并排序
speedup_list = []
if with_task:
    for seed, tasks in output.items():
        for task_num, v in tasks.items():
            elastic_pre_sjf = v.get('elastic_pre_sjf')
            elastic_pre = v.get('elastic_pre')
            sjf = v.get('sjf')
            fifo = v.get('fifo')
            if fifo and elastic_pre_sjf and elastic_pre_sjf['total_runtime'] and fifo['total_runtime']:
                speedup_total_elastic_pre_sjf_over_fifo = fifo['total_runtime'] / elastic_pre_sjf['total_runtime']
                speedup_avg_elastic_pre_sjf_over_fifo = fifo['avg_completion_time'] / elastic_pre_sjf['avg_completion_time']
                speedup_total_elastic_pre_sjf_over_sjf = sjf['total_runtime'] / elastic_pre_sjf['total_runtime']
                speedup_avg_elastic_pre_sjf_over_sjf = sjf['avg_completion_time'] / elastic_pre_sjf['avg_completion_time']
                speedup_list.append({
                    'seed': seed,
                    'task_num': task_num,
                    'total_elastic_pre_sjf_runtime': elastic_pre_sjf['total_runtime'],
                    'total_pre_runtime': elastic_pre['total_runtime'],
                    'total_sjf_runtime': sjf['total_runtime'],
                    'total_fifo_runtime': fifo['total_runtime'],
                    'avg_elastic_pre_sjf_runtime': elastic_pre_sjf['avg_completion_time'],
                    'avg_elastic_pre_runtime': elastic_pre['avg_completion_time'],
                    'avg_sjf_runtime': sjf['avg_completion_time'],
                    'avg_fifo_runtime': fifo['avg_completion_time'],
                    'speedup_total_elastic_pre_sjf_over_fifo': speedup_total_elastic_pre_sjf_over_fifo,
                    'speedup_avg_elastic_pre_sjf_over_fifo': speedup_avg_elastic_pre_sjf_over_fifo,
                    "speedup_total_elastic_pre_sjf_over_sjf": speedup_total_elastic_pre_sjf_over_sjf,
                    "speedup_avg_elastic_pre_sjf_over_sjf": speedup_avg_elastic_pre_sjf_over_sjf
                })
else:
    for seed, v in output.items():
        elastic_pre_sjf = v.get('elastic_pre_sjf')
        elastic_pre = v.get('elastic_pre')
        sjf = v.get('sjf')
        fifo = v.get('fifo')
        if fifo and elastic_pre_sjf and elastic_pre_sjf['total_runtime'] and fifo['total_runtime']:
            speedup_total_elastic_pre_sjf_over_fifo = fifo['total_runtime'] / elastic_pre_sjf['total_runtime']
            speedup_avg_elastic_pre_sjf_over_fifo = fifo['avg_completion_time'] / elastic_pre_sjf['avg_completion_time']
            speedup_total_elastic_pre_sjf_over_sjf = sjf['total_runtime'] / elastic_pre_sjf['total_runtime']
            speedup_avg_elastic_pre_sjf_over_sjf = sjf['avg_completion_time'] / elastic_pre_sjf['avg_completion_time']
            speedup_list.append({
                'seed': seed,
                'total_elastic_pre_sjf_runtime': elastic_pre_sjf['total_runtime'],
                'total_pre_runtime': elastic_pre['total_runtime'],
                'total_sjf_runtime': sjf['total_runtime'],
                'total_fifo_runtime': fifo['total_runtime'],
                'avg_elastic_pre_sjf_runtime': elastic_pre_sjf['avg_completion_time'],
                'avg_elastic_pre_runtime': elastic_pre['avg_completion_time'],
                'avg_sjf_runtime': sjf['avg_completion_time'],
                'avg_fifo_runtime': fifo['avg_completion_time'],
                'speedup_total_elastic_pre_sjf_over_fifo': speedup_total_elastic_pre_sjf_over_fifo,
                'speedup_avg_elastic_pre_sjf_over_fifo': speedup_avg_elastic_pre_sjf_over_fifo,
                "speedup_total_elastic_pre_sjf_over_sjf": speedup_total_elastic_pre_sjf_over_sjf,
                "speedup_avg_elastic_pre_sjf_over_sjf": speedup_avg_elastic_pre_sjf_over_sjf
            })

speedup_list.sort(key=lambda x: x['speedup_total_elastic_pre_sjf_over_sjf'], reverse=True)

with open(os.path.join(log_dir, 'runtime_speedup_sorted.json'), 'w') as f:
    json.dump(speedup_list, f, indent=2)

# speedup_list 只保留 speedup_total_elastic_pre_sjf_over_sjf 大于 1.3 的项
speedup_list = [item for item in speedup_list if item['speedup_total_elastic_pre_sjf_over_sjf'] > 1.3]
speedup_list.sort(key=lambda x: x['speedup_avg_elastic_pre_sjf_over_fifo'], reverse=True)

with open(os.path.join(log_dir, 'runtime_speedup_sorted_special.json'), 'w') as f:
    json.dump(speedup_list, f, indent=2)

print(f"speedup 结果已保存到 {os.path.join(log_dir, 'runtime_speedup_sorted.json')}")
print(f"speedup special 结果已保存到 {os.path.join(log_dir, 'runtime_speedup_sorted_special.json')}")
