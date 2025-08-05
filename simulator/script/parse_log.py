import re
import argparse
import json
from datetime import datetime

def parse_log(log_text):
    jobs = {}
    
    for line in log_text.split('\n'):
        # Parse job header line
        header_match = re.match(r'.*Job ID: (job_\d+) \| Name: job_\d+ \| Type: (\S+) \| Min GPUs: (\d+) \| Max GPUs: (\d+) \| Target epochs: (\d+) \| Total samples: \d+', line)
        if header_match:
            job_id, task_type, min_gpus, max_gpus, epochs = header_match.groups()
            current_job = {
                'job_id': job_id,
                'task_type': task_type,
                'epoch': int(epochs),
                'min_gpus': int(min_gpus),
                'max_gpus': int(max_gpus),
                'submit_time': None,
                'create_time': None,
                'initial_gpus': None,
                'assigned_gpus_finish_time': None,
                'finish_time': None,
                'scales': [],
            }
            jobs[job_id] = current_job
            continue
        
        # Parse submit time
        submit_match = re.match(r'.*\[(\d+\.\d+)\]: Job (job_\d+) submitted', line)
        if submit_match:
            time, job_id = submit_match.groups()
            current_job = jobs[job_id]
            current_job['submit_time'] = float(time)
            continue
            
        # Parse start time and GPUs
        start_match = re.match(r'.*\[(\d+\.\d+)\]: Job (job_\d+) starting job execution, in \[(.*)\]', line)
        if start_match:
            time, job_id, gpus_str = start_match.groups()
            current_job = jobs[job_id]
            current_job['create_time'] = float(time)
            # Parse GPU list
            gpus = []
            for gpu in re.findall(r'\((\d+), (\d+)\)', gpus_str):
                gpus.append([int(gpu[0]), int(gpu[1])])
            current_job['initial_gpus'] = gpus
            continue
            
        # Parse assigned GPUs finish time
        gpus_finish_match = re.match(r'.*\[(\d+\.\d+)\]: Job (job_\d+) finished waiting for assigned gpus', line)
        if gpus_finish_match:
            time, job_id = gpus_finish_match.groups()
            current_job = jobs[job_id]
            current_job['assigned_gpus_finish_time'] = float(time)
            continue
            
        # Parse finished time
        finish_match = re.match(r'.*\[(\d+\.\d+)\]: Job (job_\d+) finished', line)
        if finish_match:
            time, job_id = finish_match.groups()
            current_job = jobs[job_id]
            current_job['finish_time'] = float(time)
            continue
            
        # Parse actual scale operation
        scale_op_match = re.match(r'.*\[(\d+\.\d+)\]: Job (job_\d+) scale from \[(.*)\] to \[(.*)\]', line)
        if scale_op_match:
            time, job_id, from_gpus_str, to_gpus_str = scale_op_match.groups()
            current_job = jobs[job_id]
            
            # Parse to_gpus (the new GPU allocation)
            from_gpus = []
            for gpu in re.findall(r'\((\d+), (\d+)\)', from_gpus_str):
                from_gpus.append([int(gpu[0]), int(gpu[1])])
                    
            # Parse to_gpus
            to_gpus = []
            for gpu in re.findall(r'\((\d+), (\d+)\)', to_gpus_str):
                to_gpus.append([int(gpu[0]), int(gpu[1])])
                
            current_job['scales'].append([float(time), to_gpus])
            continue
            
    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse job log file")
    parser.add_argument('log_file', type=str, help='Path to the log file to parse')
    parser.add_argument('--output', type=str, default='output.json', help='Path to output JSON file')
    args = parser.parse_args()
    
    with open(args.log_file, 'r') as file:
        log_text = file.read()
    
    jobs = parse_log(log_text)
    
    # Write to JSON file
    with open(args.output, 'w') as f:
        json.dump(jobs, f, indent=2)
    print(f"Successfully parsed and saved to {args.output}")
        
    for job_id, job_info in jobs.items():
        print(f"{job_info},")
    
    