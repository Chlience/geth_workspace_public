import argparse
import random

from loguru import logger

from geth.base.operation import SystemOperation
from geth.base.zmq_link import ZmqClient


def create_parser():
    # Create the parser
    parser = argparse.ArgumentParser(description="Geth Client commandline.")

    # Add arguments
    parser.add_argument(
        "-a",
        "--address",
        type=str,
        help="Address of hub to connect to",
        default="localhost",
    )
    parser.add_argument(
        "-p", "--port", type=int, help="Port of hub to connect to", default=8899
    )

    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for the 'list_agent' command
    parser_list_agents = subparsers.add_parser(  # noqa: F841
        "list_agent", help="List available agents registered with hub"
    )

    # Subparser for the 'show_agent' command
    parser_show_agent = subparsers.add_parser(
        "show_agent", help="Show detailed agent information"
    )
    parser_show_agent.add_argument(
        "agent_id", type=int, help="Agent ID of target agent"
    )

    # Subparser for the 'list_task' command
    parser_list_tasks = subparsers.add_parser("list_task", help="List all tasks")  # noqa: F841

    # Subparser for the 'show_task' command
    parser_show_task = subparsers.add_parser(
        "show_task", help="Show detailed task information"
    )
    parser_show_task.add_argument("task_id", type=int, help="Task ID")

    # Subparser for the 'create_task' command
    parser_create_task = subparsers.add_parser("create_task", help="Create a new task")
    parser_create_task.add_argument(
        "task_file_path", type=str, help="Task description file"
    )
    # a list of agents seperated by ",", parse it by parser
    parser_create_task.add_argument(
        "agent_ids", type=str, help="List of agent IDs seperated by ','"
    )

    # Subparser for the 'update_task' command
    parser_update_task = subparsers.add_parser(
        "update_task", help="Update task resource and running status"
    )
    parser_update_task.add_argument("task_id", type=int, help="Task ID")
    parser_update_task.add_argument(
        "--agent_ids",
        required=False,
        type=str,
        default=None,
        help="List of agent IDs seperated by ','",
    )

    # Subparser for the 'delete_task' command
    parser_delete_task = subparsers.add_parser("delete_task", help="Delete a task")
    parser_delete_task.add_argument("task_id", type=int, help="Task ID")

    return parser


def main():
    parser = create_parser()

    # Parse arguments
    args = parser.parse_args()

    zmq_endpoint = "tcp://{}:{}".format(args.address, args.port)
    zmq_name = "client-" + hex(random.randint(0, 2**16))
    zmq_client = ZmqClient(zmq_name)
    zmq_client.connect(zmq_endpoint)
    logger.info(f"Client {zmq_name} connect to GethHub at {zmq_endpoint}")

    match args.command:
        case "list_agent":
            op = SystemOperation(SystemOperation.OpCode.ListAgent, ())
        case "show_agent":
            op = SystemOperation(SystemOperation.OpCode.ShowAgent, (args.agent_id,))
        case "list_task":
            op = SystemOperation(SystemOperation.OpCode.ListTask, ())
        case "show_task":
            op = SystemOperation(SystemOperation.OpCode.ShowTask, (args.task_id,))
        case "create_task":
            agent_ids = args.agent_ids.split(",")
            op = SystemOperation(
                SystemOperation.OpCode.CreateTask, (args.task_file_path, agent_ids)
            )
        case "update_task":
            agent_ids = args.agent_ids.split(",") if args.agent_ids else None
            op = SystemOperation(
                SystemOperation.OpCode.UpdateTask,
                (args.task_id, agent_ids),
            )
        case "delete_task":
            op = SystemOperation(SystemOperation.OpCode.DeleteTask, (args.task_id,))
        case _:
            raise Exception("No such command")

    zmq_client.send(op)
    response = zmq_client.recv()
    logger.debug(f"Response from hub: {response}")


if __name__ == "__main__":
    main()
