import os
import signal
import subprocess
import sys
import time
from typing import List, Tuple, Dict
from loguru import logger

from geth.base.operation import SystemOperation
from geth.base.zmq_link import ZmqClient


def setup_system(device_num: int, port: int = 8900, agent_port: List[int] = None) -> Tuple[subprocess.Popen, List[subprocess.Popen]]:
    if agent_port == None:
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


if __name__ == "__main__":
    hub_proc, agent_procs = setup_system(8)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--end_size", type=int, default=8)
    parser.add_argument("--wait_time", type=int, default=25)
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()

    try:
        time.sleep(2)

        zmq_endpoint = "tcp://{}:{}".format("localhost", 8900)
        zmq_name = "client-system"
        zmq_client = ZmqClient(zmq_name)
        zmq_client.connect(zmq_endpoint)
        logger.info("Client connect to GethHub")

        op = SystemOperation(
            SystemOperation.OpCode.CreateTask,
            (
                f"/workspace/geth/examples/{args.model_name}.py",
                list(range(args.start_size)),
            ),
        )
        logger.info(f"Requesting: {op}")
        zmq_client.send(op)
        response = zmq_client.recv()
        logger.info(f"Response from hub: {response}")
        time.sleep(args.wait_time)

        op = SystemOperation(
            SystemOperation.OpCode.UpdateTask, (0, list(range(args.end_size)))
        )
        logger.info(f"Requesting: {op}")
        zmq_client.send(op)
        response = zmq_client.recv()
        logger.info(f"Response from hub: {response}")
        time.sleep(args.wait_time)

        del zmq_client
    except Exception as e:
        print(e)

    for agent_proc in agent_procs:
        try:
            print(agent_proc.pid)
            os.kill(agent_proc.pid, signal.SIGTERM)
        except Exception as e:
            print(e)
    try:
        print(hub_proc.pid)
        os.kill(hub_proc.pid, signal.SIGTERM)
    except Exception as e:
        print(e)

    try:
        os.system("pkill -f python3")
    except Exception as e:
        print(e)
