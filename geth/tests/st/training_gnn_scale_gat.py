import os
import signal
import subprocess
import sys
import time

from loguru import logger

from geth.base.operation import SystemOperation
from geth.base.zmq_link import ZmqClient


def setup_system(device_num: int, port: int = 8899):
    hub_proc = subprocess.Popen(
        ["python3", "geth/utils/hub_cli.py", f"--port={port}"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    agent_procs = [
        subprocess.Popen(
            [
                "python3",
                "geth/utils/agent_cli.py",
                f"{i}",
                f"{1 - i}",
                "--address=localhost",
                f"--port={port}",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        for i in range(device_num)
    ]
    return hub_proc, agent_procs


if __name__ == "__main__":
    hub_proc, agent_procs = setup_system(2)

    # hub_proc.terminate()
    # for agent_proc in agent_procs:
    #     agent_proc.terminate()
    # hub_proc.wait()
    # for agent_proc in agent_procs:
    #     agent_proc.wait()

    try:
        time.sleep(2)

        zmq_endpoint = "tcp://{}:{}".format("localhost", 8899)
        zmq_name = "client-system"
        zmq_client = ZmqClient(zmq_name)
        zmq_client.connect(zmq_endpoint)
        logger.info("Client connect to GethHub")

        # todo: start with 2 will cause crash due to assert that rank 0 will always be rank 0 not satisified
        op = SystemOperation(
            SystemOperation.OpCode.CreateTask, ("./examples/gat_pubmed.py", [0, 1])
        )
        logger.info(f"Requesting: {op}")
        zmq_client.send(op)
        response = zmq_client.recv()
        logger.info(f"Response from hub: {response}")
        time.sleep(3)

        op = SystemOperation(SystemOperation.OpCode.UpdateTask, (0, [0]))
        logger.info(f"Requesting: {op}")
        zmq_client.send(op)
        response = zmq_client.recv()
        logger.info(f"Response from hub: {response}")
        time.sleep(1)

        op = SystemOperation(SystemOperation.OpCode.UpdateTask, (0, [0, 1]))
        logger.info(f"Requesting: {op}")
        zmq_client.send(op)
        response = zmq_client.recv()
        logger.info(f"Response from hub: {response}")
        time.sleep(1)

        op = SystemOperation(SystemOperation.OpCode.UpdateTask, (0, [0]))
        logger.info(f"Requesting: {op}")
        zmq_client.send(op)
        response = zmq_client.recv()
        logger.info(f"Response from hub: {response}")
        time.sleep(1)

        op = SystemOperation(SystemOperation.OpCode.UpdateTask, (0, [0, 1]))
        logger.info(f"Requesting: {op}")
        zmq_client.send(op)
        response = zmq_client.recv()
        logger.info(f"Response from hub: {response}")
        time.sleep(1)

        del zmq_client
    except Exception as e:
        print(e)

    try:
        input("Press any key to terminate...\n")
    except Exception as e:
        print(e)
        print("Terminating...")

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
        os.system("pkill -f 'python3 geth/utils/agent_cli.py'")
    except Exception as e:
        print(e)
