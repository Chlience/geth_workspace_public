from dataclasses import dataclass
from typing import List, Union, Optional
import random
import numpy as np
import pandas as pd 
from datetime import datetime, timedelta

@dataclass
class TaskRandomSourceConfig:
    task_type: str
    samples: Union[List[int], int]
    epochs: List[int]
    batch_size: int
    init_gpus: List[int]

RANDOM_SOURCE = [
    TaskRandomSourceConfig("gcn_reddit", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    TaskRandomSourceConfig("gcn_yelp", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    TaskRandomSourceConfig("gcn_ogbn-products", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    # TaskRandomSourceConfig("gcn_ogbn-arxiv", [1], [500, 800], 1, [1]),
    TaskRandomSourceConfig("gat_reddit", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    TaskRandomSourceConfig("gat_yelp", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    TaskRandomSourceConfig("gat_ogbn-products", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    # TaskRandomSourceConfig("gat_ogbn-arxiv", [1], [500, 800], 1, [1]),
    TaskRandomSourceConfig("gin_reddit", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    TaskRandomSourceConfig("gin_yelp", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    TaskRandomSourceConfig("gin_ogbn-products", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    # TaskRandomSourceConfig("gin_ogbn-arxiv", [1], [500, 800], 1, [1]),
    TaskRandomSourceConfig("sage_reddit", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    TaskRandomSourceConfig("sage_yelp", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    TaskRandomSourceConfig("sage_ogbn-products", [1], [500, 800, 1500], 1, [1, 2, 4, 6, 8]),
    # TaskRandomSourceConfig("sage_ogbn-arxiv", [1], [500, 800], 1, [1]),
]

@dataclass
class TaskConfig:
    task_type: str
    samples: int
    epochs: int
    batch_size: int
    init_gpus: int
    arrive_time: float

def generate_random_trace(n, random_source, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    trace = []
    for _ in range(n):
        src = random.choice(random_source)
        task_type = src.task_type
        if _ % 100 == 0:
            task_type = "gin_ogbn-arxiv"
        samples = random.choice(src.samples) if isinstance(src.samples, list) else src.samples
        epochs = random.choice(src.epochs) if isinstance(src.epochs, list) else src.epochs
        batch_size = random.choice(src.batch_size) if isinstance(src.batch_size, list) else src.batch_size
        init_gpus = random.choice(src.init_gpus) if isinstance(src.init_gpus, list) else src.init_gpus
        arrive_time = 0.0
        trace.append(TaskConfig(task_type, samples, epochs, batch_size, init_gpus, arrive_time))
    return trace

def generate_nonhomogeneous_poisson_timestamps(lambda_base, n, start_time=-1, seed: Optional[int] = None, mode="exp"):
    """
    生成服从泊松过程的时间戳
    
    参数:
    lambda_base: 基础强度
    n: 生成的事件数量
    start_time: 起始时间（默认为当前时间）
    
    返回:
    timestamps: 时间戳列表（Unix时间戳，单位为秒）
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if start_time == -1 and mode == "exp":
        start_time = random.randint(0, 48) / 2 * 3600
    elif mode == "demo":
        start_time = 0
    print(f"start time: {start_time}")
    
    # 定义时变强度函数
    # def intensity_function(t_hours):
    #     """t_hours是一天中的小时数(0-24)"""
    #     return lambda_base + lambda_amplitude * np.sin(np.pi * (t_hours - 12) / 12)**2
    
    def work_hours_intensity_exp(t_hours):
        """工作时间模式：工作时间高，非工作时间低，非工作时间平滑过渡"""
        if 10 <= t_hours < 19:  # 工作时间
            return lambda_base
        elif 0 <= t_hours < 10:  # 早晨，正弦波上升
            # 0点为最低，9点为最高
            return lambda_base * (0.1 + 0.9 * np.sin((np.pi / 2) * (t_hours / 10)))
        elif 19 <= t_hours < 24:  # 晚上，正弦波下降
            # 21点为最高，24点为最低
            return lambda_base * (0.1 + 0.9 * np.sin((np.pi / 2) * ((24 - t_hours) / 5)))
        else:  # 其他时间
            return lambda_base * 0.1  # 夜间很低的强度
        
    def work_hours_intensity_demo(t_hours):
        return lambda_base
    
    work_hours_intensity = work_hours_intensity_exp if mode == "exp" else work_hours_intensity_demo
    
    # 找出强度函数的上界
    lambda_max = lambda_base
    
    timestamps = []
    current_time = start_time
    
    # 使用薄化法(Thinning)生成非齐次泊松过程
    while len(timestamps) < n:
        # 生成下一个事件的时间间隔（指数分布）
        dt = np.random.exponential(scale=1.0/lambda_max)
        
        # 更新当前时间
        current_time += dt
        
        # 获取当前时间在一天中的小时数
        current_hour = (current_time % 86400) / 3600
        
        # 计算当前时间点的强度
        lambda_t = work_hours_intensity(current_hour)
        
        # 以λ(t)/λ_max的概率接受该事件
        if np.random.random() <= lambda_t / lambda_max:
            timestamps.append(current_time - start_time) # generate relative time
    return timestamps

trace_csv_path = "/workspace/simulator/pai.csv"
# trace_csv_path = "/workspace/simulator/ms.csv"
trace_csv = pd.read_csv(trace_csv_path, parse_dates=["timestamp"])
trace_csv = trace_csv.sort_values("timestamp")

def generate_timestamps(num_tasks, start=0, duration=24, seed=0):
    # 构建trace文件路径，使用philly.csv作为数据源
    # 使用 pandas 读取CSV文件，并解析 timestamp 列为日期时间格式
    # 数据预处理：过滤掉持续时间小于 60 秒的记录
    # trace = trace[trace.duration >= 60]
    # 过滤掉GPU时间超过 1000 小时的记录
    # trace = trace[trace.gpu_time < 1000 * 3600]
    # 按时间排序 trace
    # 保留每天 start 点到 start + duration 点之间的数据
    new_trace_csv = trace_csv[(trace_csv.timestamp.dt.hour >= start) &
                  (trace_csv.timestamp.dt.hour < start + duration)]
    sample = new_trace_csv.sample(n=num_tasks, random_state=seed)
    
    timestamps = [timedelta(hours=task.timestamp.hour - start,
                            minutes=task.timestamp.minute,
                            seconds=task.timestamp.second).total_seconds()
                  for task in sample.itertuples()]
    
    return timestamps


def combine_trace_and_timestamps(trace, timestamps):
    """
    Combine a list of TaskConfig (trace) and a list of timestamps.
    Sets each TaskConfig's arrive_time to the corresponding timestamp.
    Returns the updated trace.
    """
    assert len(trace) == len(timestamps), "Trace and timestamps must have the same length."
    for task, ts in zip(trace, timestamps):
        task.arrive_time = ts
    return trace

def generate_trace_with_timestamps(num_tasks, seed: Optional[int] = None, scale: Optional[float] = 1):
    trace = generate_random_trace(num_tasks, random_source=RANDOM_SOURCE, seed=seed)
    # 目前使用静态的 11 点到 12 点之间的数据
    time_stamps = generate_timestamps(num_tasks=num_tasks, start=11, duration=1, seed=seed)
    if scale is not None:
        time_stamps = [ts * scale for ts in time_stamps]
    combined_trace = combine_trace_and_timestamps(trace, time_stamps)
    return combined_trace

if __name__ == "__main__":
    combined_trace = generate_trace_with_timestamps(2, seed=42, scale=0.5)
