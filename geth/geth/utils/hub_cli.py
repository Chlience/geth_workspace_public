import argparse

from loguru import logger

from geth.hub.hub import GethHub


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Geth Hub commandline.")

    # Add arguments
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        required=False,
        help="Port to listen to",
        default=8899,
    )

    # Parse arguments
    args = parser.parse_args()

    logger.info("GethHub starting...")

    # Create the agent
    hub = GethHub(args.port)
    hub.execute()


if __name__ == "__main__":
    main()
