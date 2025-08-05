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
    tasks = [
        {'job_id': 'job_0', 'task_type': 'sage_ogbn-products', 'epoch': 1500, 'min_gpus': '2', 'max_gpus': '8', 'submit_time': 6.0, 'create_time': 6.0, 'initial_gpus': [[0, 2], [0, 3]], 'assigned_gpus_finish_time': 30.56, 'finish_time': 340.92, 'scales': []},
        {'job_id': 'job_6', 'task_type': 'gin_yelp', 'epoch': 800, 'min_gpus': '2', 'max_gpus': '8', 'submit_time': 6.0, 'create_time': 6.0, 'initial_gpus': [[0, 0], [0, 1]], 'assigned_gpus_finish_time': 14.26, 'finish_time': 138.0, 'scales': [[20.0, [[0, 0], [0, 1], [0, 4], [0, 5], [0, 6], [0, 7]]]]},
        {'job_id': 'job_2', 'task_type': 'gcn_reddit', 'epoch': 1500, 'min_gpus': '6', 'max_gpus': '8', 'submit_time': 10.8, 'create_time': 124.0, 'initial_gpus': [[0, 0], [0, 1], [0, 4], [0, 5], [0, 6], [0, 7]], 'assigned_gpus_finish_time': 150.26, 'finish_time': 359.53, 'scales': []},
        {'job_id': 'job_17', 'task_type': 'gin_ogbn-products', 'epoch': 500, 'min_gpus': '6', 'max_gpus': '8', 'submit_time': 301.2, 'create_time': 345.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]], 'assigned_gpus_finish_time': 367.58, 'finish_time': 389.8, 'scales': []},
        {'job_id': 'job_1', 'task_type': 'gin_reddit', 'epoch': 500, 'min_gpus': '6', 'max_gpus': '8', 'submit_time': 372.6, 'create_time': 375.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]], 'assigned_gpus_finish_time': 400.69, 'finish_time': 462.89, 'scales': []},
        {'job_id': 'job_15', 'task_type': 'sage_ogbn-products', 'epoch': 500, 'min_gpus': '6', 'max_gpus': '8', 'submit_time': 468.0, 'create_time': 468.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]], 'assigned_gpus_finish_time': 489.49, 'finish_time': 535.71, 'scales': [[495.0, [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]]]]},
        {'job_id': 'job_13', 'task_type': 'gin_ogbn-products', 'epoch': 500, 'min_gpus': '1', 'max_gpus': '8', 'submit_time': 617.4, 'create_time': 618.0, 'initial_gpus': [[0, 0]], 'assigned_gpus_finish_time': 642.89, 'finish_time': 674.47, 'scales': [[648.0, [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]]]]},
        {'job_id': 'job_14', 'task_type': 'gcn_yelp', 'epoch': 1500, 'min_gpus': '8', 'max_gpus': '8', 'submit_time': 910.2, 'create_time': 911.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]], 'assigned_gpus_finish_time': 923.2, 'finish_time': 1137.86, 'scales': []},
        {'job_id': 'job_11', 'task_type': 'gin_reddit', 'epoch': 500, 'min_gpus': '4', 'max_gpus': '8', 'submit_time': 1086.6, 'create_time': 1123.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3]], 'assigned_gpus_finish_time': 1149.19, 'finish_time': 1204.18, 'scales': []},
        {'job_id': 'job_18', 'task_type': 'gcn_reddit', 'epoch': 1500, 'min_gpus': '6', 'max_gpus': '8', 'submit_time': 1093.2, 'create_time': 1579.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]], 'assigned_gpus_finish_time': 1605.26, 'finish_time': 1814.53, 'scales': []},
        {'job_id': 'job_10', 'task_type': 'sage_yelp', 'epoch': 800, 'min_gpus': '6', 'max_gpus': '8', 'submit_time': 1179.6, 'create_time': 1190.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]], 'assigned_gpus_finish_time': 1204.18, 'finish_time': 1400.68, 'scales': [[1210.0, [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]]]]},
        {'job_id': 'job_4', 'task_type': 'sage_ogbn-products', 'epoch': 1500, 'min_gpus': '2', 'max_gpus': '8', 'submit_time': 1332.0, 'create_time': 1748.0, 'initial_gpus': [[0, 6], [0, 7]], 'assigned_gpus_finish_time': 1772.57, 'finish_time': 2082.91, 'scales': []},
        {'job_id': 'job_9', 'task_type': 'sage_yelp', 'epoch': 800, 'min_gpus': '8', 'max_gpus': '8', 'submit_time': 1341.6, 'create_time': 1386.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]], 'assigned_gpus_finish_time': 1400.68, 'finish_time': 1593.76, 'scales': []},
        {'job_id': 'job_12', 'task_type': 'gat_reddit', 'epoch': 800, 'min_gpus': '8', 'max_gpus': '8', 'submit_time': 1435.2, 'create_time': 3910.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]], 'assigned_gpus_finish_time': 3936.02, 'finish_time': 4562.79, 'scales': []},
        {'job_id': 'job_8', 'task_type': 'gin_ogbn-products', 'epoch': 1500, 'min_gpus': '1', 'max_gpus': '8', 'submit_time': 1473.6, 'create_time': 1579.0, 'initial_gpus': [[0, 6]], 'assigned_gpus_finish_time': 1603.89, 'finish_time': 1762.0, 'scales': [[1609.0, [[0, 6], [0, 7]]]]},
        {'job_id': 'job_3', 'task_type': 'gat_reddit', 'epoch': 1500, 'min_gpus': '2', 'max_gpus': '8', 'submit_time': 1633.2, 'create_time': 1800.0, 'initial_gpus': [[0, 4], [0, 5]], 'assigned_gpus_finish_time': 1826.53, 'finish_time': 3105.78, 'scales': [[2015.0, [[0, 4], [0, 5], [0, 0], [0, 1], [0, 2], [0, 3]]], [2073.0, [[0, 4], [0, 5], [0, 0], [0, 1], [0, 2], [0, 3], [0, 7]]]]},
        {'job_id': 'job_7', 'task_type': 'sage_reddit', 'epoch': 500, 'min_gpus': '4', 'max_gpus': '8', 'submit_time': 1687.8, 'create_time': 1800.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3]], 'assigned_gpus_finish_time': 1826.22, 'finish_time': 1919.84, 'scales': []},
        {'job_id': 'job_5', 'task_type': 'gcn_ogbn-products', 'epoch': 500, 'min_gpus': '8', 'max_gpus': '8', 'submit_time': 1761.6, 'create_time': 3884.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]], 'assigned_gpus_finish_time': 3905.92, 'finish_time': 3924.73, 'scales': []},
        {'job_id': 'job_19', 'task_type': 'sage_reddit', 'epoch': 500, 'min_gpus': '4', 'max_gpus': '8', 'submit_time': 1831.2, 'create_time': 1905.0, 'initial_gpus': [[0, 0], [0, 1], [0, 2], [0, 3]], 'assigned_gpus_finish_time': 1931.22, 'finish_time': 2024.84, 'scales': []},
        {'job_id': 'job_16', 'task_type': 'gat_reddit', 'epoch': 1500, 'min_gpus': '1', 'max_gpus': '8', 'submit_time': 2054.4, 'create_time': 2069.0, 'initial_gpus': [[0, 6]], 'assigned_gpus_finish_time': 2093.78, 'finish_time': 3898.72, 'scales': [[3097.0, [[0, 6], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 7]]]]},
    ] # seed 26
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