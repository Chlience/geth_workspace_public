import os
import signal
import subprocess
import sys
import time
from typing import List, Tuple, Dict
from loguru import logger

from geth.base.operation import SystemOperation
from geth.base.zmq_link import ZmqClient

import threading
import argparse

def setup_system(device_num: int, port: int = 8900, agent_port: List[int] = []) -> Tuple[subprocess.Popen, List[subprocess.Popen]]:
    if len(agent_port) == 0:
        agent_port = [port + 1 + i * 2 for i in range(device_num)]
    assert device_num == len(agent_port), "Number of devices must match the number of agent ports provided."
    # hub 控制中心
    hub_proc = subprocess.Popen(
        ["python3", "geth/utils/hub_cli.py", f"--port={port}"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    # agent 受控制进程，worker
    agent_procs = [
        subprocess.Popen(
            [
                "python3",
                "geth/utils/agent_cli.py",
                f"{i}",
                f"{i}",
                "--address=localhost",
                f"--port={port}",
                f"--agent_port={agent_port[i]}",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        for i in range(device_num)
    ]
    return hub_proc, agent_procs

barrier = None

def run_task(task: Dict) -> None:
    # 设置进程环境
    hub_proc, agent_procs = setup_system(8, task['port'], task['agent_port'])
    time.sleep(2)
    
    zmq_endpoint = "tcp://{}:{}".format("localhost", task['port'])
    zmq_name = "client-system"
    zmq_client = ZmqClient(zmq_name)
    zmq_client.connect(zmq_endpoint)
    
    global barrier
    barrier.wait()
    
    begin_time = time.time()
    logger.debug(f"[{time.time() - begin_time:>8.3f}]: {task}")
    
    sleep_time = task["create_time"] - (time.time() - begin_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    logger.debug(f"[{time.time() - begin_time:>8.3f}]: Task {task['task_type']} on port {task['port']} start at {task['create_time']}.")
    op = SystemOperation(
        SystemOperation.OpCode.CreateTask,
        (
            f"/workspace/geth/examples/{task['task_type']}.py",
            task['initial_gpus'],
            task['epoch'],
            task['job_id'],
        ),
    )
    logger.info(f"Requesting: {op}")
    zmq_client.send(op)
    response = zmq_client.recv()
    logger.info(f"Response from hub: {response}")
    
    for scale_time, scale_gpus in task["scales"]:
        sleep_time = scale_time - (time.time() - begin_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        logger.debug(f"[{time.time() - begin_time:>8.3f}]: Task {task['task_type']} on port {task['port']} scaling to {scale_gpus} at {scale_time:.3f} seconds.")
        op = SystemOperation(
            SystemOperation.OpCode.UpdateTask, (0, scale_gpus)
        )
        logger.info(f"Requesting: {op}")
        zmq_client.send(op)
        response = zmq_client.recv()
        logger.info(f"Response from hub: {response}")
        
    del zmq_client
    
    sleep_time = task["finish_time"]  - (time.time() - begin_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    logger.debug(f"[{time.time() - begin_time:>8.3f}]: Task {task['task_type']} on port {task['port']} should shutdown at {task['finish_time']:.3f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize task execution timeline from log file.')
    parser.add_argument('log_file', help='Path to the log file')
    args = parser.parse_args()
    # Load jobs from the log file
    
    import json
    with open(args.log_file, 'r') as f:
        tasks = json.load(f)
    
    num_gpus = 8
    port_per_agent = 2
    # use port + 1 for recover dist group
    for id, task in enumerate(tasks):
        task.update({"port": 8900 + id})
        base_port = 32000 + id * num_gpus * port_per_agent
        task.update({"agent_port": list(range(base_port, base_port + num_gpus * port_per_agent, 2))})
        task.update({"initial_gpus": [gpu_info[1] for gpu_info in task["initial_gpus"]]})
        scales = []
        for scale in task["scales"]:
            scales.append([scale[0], [gpu_info[1] for gpu_info in scale[1]]])
        task.update({"scales": scales})
        
    barrier = threading.Barrier(len(tasks) + 1)
    threads = []
    for i in range(len(tasks)):
        t = threading.Thread(target=run_task, args=(tasks[i], ))
        t.start()
        threads.append(t)
    barrier.wait()

    logger.info("All tasks are ready, starting execution...")
    
    for t in threads:
        t.join()
    
    logger.info("All tasks should be completed, shutting down...")