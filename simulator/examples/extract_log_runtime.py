import os
import re
import json
import argparse

parser = argparse.ArgumentParser(description='Extract runtime and speedup from simulation logs.')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory containing the log files.')
args = parser.parse_args()
log_dir = args.log_dir

output = {}

pattern = re.compile(r'(elasgnn_|sjf_|yarn-cs_|)?simulation_(\d+)\.log')
runtime_pattern = re.compile(r'Total Runtime: ([\d.]+) seconds')
avg_pattern = re.compile(r'Avg Completion Time: ([\d.]+) seconds')

with_task = False
for fname in os.listdir(log_dir):
    m = pattern.fullmatch(fname)
    if m:
        print("Processing task log:", fname)
        prefix = m.group(1) or ''
        rank = m.group(2)
    else:
        continue
    if prefix == 'elasgnn_':
        log_type = 'elasgnn'
    elif prefix == 'sjf_':
        log_type = 'sjf'
    else:
        log_type = 'yarn'
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
    if rank not in output:
        output[rank] = {}
    output[rank][log_type] = {
        'total_runtime': total_runtime,
        'avg_completion_time': avg_completion_time
    }

# with open(os.path.join(log_dir, 'runtime_summary.json'), 'w') as f:
#     json.dump(output, f, indent=2)

# 计算speedup并排序
speedup_list = []
for rank, v in output.items():
    elasgnn = v.get('elasgnn')
    sjf = v.get('sjf')
    yarn = v.get('yarn')
    if yarn and elasgnn and elasgnn['total_runtime'] and yarn['total_runtime']:
        speedup_total_elasgnn_over_yarn = yarn['total_runtime'] / elasgnn['total_runtime']
        speedup_avg_elasgnn_over_yarn = yarn['avg_completion_time'] / elasgnn['avg_completion_time']
        speedup_total_elasgnn_over_sjf = sjf['total_runtime'] / elasgnn['total_runtime']
        speedup_avg_elasgnn_over_sjf = sjf['avg_completion_time'] / elasgnn['avg_completion_time']
        speedup_list.append({
            'rank': rank,
            'total_elasgnn_runtime': elasgnn['total_runtime'],
            'total_sjf_runtime': sjf['total_runtime'],
            'total_yarn_runtime': yarn['total_runtime'],
            'avg_elasgnn_runtime': elasgnn['avg_completion_time'],
            'avg_sjf_runtime': sjf['avg_completion_time'],
            'avg_yarn_runtime': yarn['avg_completion_time'],
            'speedup_total_elasgnn_over_yarn': speedup_total_elasgnn_over_yarn,
            'speedup_avg_elasgnn_over_yarn': speedup_avg_elasgnn_over_yarn,
            "speedup_total_elasgnn_over_sjf": speedup_total_elasgnn_over_sjf,
            "speedup_avg_elasgnn_over_sjf": speedup_avg_elasgnn_over_sjf
        })

speedup_list.sort(key=lambda x: x['speedup_total_elasgnn_over_sjf'], reverse=True)

# with open(os.path.join(log_dir, 'runtime_speedup_sorted.json'), 'w') as f:
#     json.dump(speedup_list, f, indent=2)

# speedup_list 只保留 speedup_total_elasgnn_over_sjf 大于 1.3 的项
speedup_list = [item for item in speedup_list if item['speedup_total_elasgnn_over_sjf'] > 1.3]
speedup_list.sort(key=lambda x: x['speedup_avg_elasgnn_over_yarn'], reverse=True)

with open(os.path.join(log_dir, 'runtime_speedup.json'), 'w') as f:
    json.dump(speedup_list, f, indent=2)

print(f"Results has been saved to {os.path.join(log_dir, 'runtime_speedup.json')}")
