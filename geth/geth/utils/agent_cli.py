import argparse
import subprocess

from loguru import logger

from geth.agent.agent import GethAgent


def get_default_ip():
    output = subprocess.check_output(
        ["ip", "-o", "route", "get", "1.1.1.1"], universal_newlines=True
    )
    return output.split(" ")[6]


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Geth Agent commandline.")

    # Add arguments
    parser.add_argument("agent_id", type=str, help="Agent ID")
    parser.add_argument("device_id", type=int, help="Device ID")
    parser.add_argument(
        "-a", "--address", type=str, required=True, help="Address of hub to connect to"
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port of hub to connect to",
        default=8899,
    )
    parser.add_argument(
        "--agent_port",
        type=int,
        required=True,
        default=32000,
    )

    # Parse arguments
    args = parser.parse_args()

    logger.info("Agent [{agent_id}] starting", agent_id=args.agent_id)
    zmq_endpoint = "tcp://{}:{}".format(args.address, args.port)
    logger.info("Zmq Endpoint at {zmq_endpoint}", zmq_endpoint=zmq_endpoint)

    ip_address = get_default_ip()
    logger.info("IP Address: {ip_address}", ip_address=ip_address)

    # Create the agent
    port = args.agent_port
    agent = GethAgent(args.agent_id, args.device_id, ip_address, port, zmq_endpoint)
    agent.execute()


if __name__ == "__main__":
    main()
