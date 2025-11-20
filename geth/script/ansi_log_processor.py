import strip_ansi
from datetime import datetime, timedelta
import re
import argparse

def process_log_file(log_text):
    # 编译正则表达式来匹配日志行中的时间戳
    time_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})')
    begin_pattern = re.compile(r'__main__:run_task:60')
    
    exclude_patterns = [
        'geth.base.zmq_link',
        'geth.agent.agent:register',
        '__main__:main',
        'created with device',
        'OpCode.RegisterAgent',
        'Registered agent AgentInfo',
        'so path is ',
        'UserWarning: ',
        '################################################################################',
        'WARNING!',
        'The \'datapipes\', \'dataloader2\' modules are deprecated and will be removed in a',
        'future torchdata release! Please see https://github.com/pytorch/data/issues/1196',
        'to learn more and leave feedback.',
        '################################################################################',
        'deprecation_warning()',
        'WARNING: Logging before InitGoogleLogging() is written to STDERR',
        '>>PrePart took',
        'Assign nodes to local partitions',
        'Find remote nodes',
        'Remove duplicates',
        'Build mappings',
        '>>PreBuild map took',
        '>>PreBuild request took',
        '[info] edges_chunk finished',
        '[info] edges insert finished',
        '[info] graph creation finished',
        'Number of cross edges',
        '>>PreBuild subgraph took',
        'Prepartition took',
        'Serialize took',
        'Prepartition all took',
        'Prepartition overall took',
    ]
    
    for line in log_text.splitlines():
        match = begin_pattern.search(line)
        if match:
            match = time_pattern.search(line)
            if not match:
                print(f"Warning: No timestamp found in line: {line.strip()}")
                exit(0)
            original_time_str = match.group(1)
            # 将日志时间转换为datetime对象
            ref_time = datetime.strptime(original_time_str, "%Y-%m-%d %H:%M:%S.%f")
            break
    
    output_lines = []
    
    for line in log_text.splitlines():
        if line == '\n':
            continue
        if any(pattern in line for pattern in exclude_patterns):
            continue  # 跳过这些行
        # 查找时间戳
        match = time_pattern.search(line)
        if match:
            original_time_str = match.group(1)
            # 将日志时间转换为datetime对象
            log_time = datetime.strptime(original_time_str, "%Y-%m-%d %H:%M:%S.%f")
            # 计算时间差
            time_diff = log_time - ref_time
            # 替换原始时间戳为时间差
            new_line = time_pattern.sub(f"[{time_diff.total_seconds():.3f}]", line)
            output_lines.append(new_line)
    
    return "\n".join(output_lines)

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a log file to remove ANSI escape sequences and calculate time differences.")
    parser.add_argument('log_file', type=str, help='Path to the input log file')
    args = parser.parse_args()
    
    input_file = args.log_file
    output_file = input_file.replace('.log', '_processed.log')
    
    with open(input_file, 'r') as f:
        text = f.read()
    
    # 清理ANSI转义序列s
    clean_text = strip_ansi.strip_ansi(text)
    
    output_text = process_log_file(clean_text)
    
    with open(output_file, 'w') as f:
        f.write(output_text)
    
    print(f"Processed log saved to {output_file}")