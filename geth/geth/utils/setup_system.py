import os
import signal
import subprocess
import sys


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
