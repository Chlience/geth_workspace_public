from enum import Enum


class AgentStatus(Enum):
    IDLE = 1
    BUSY = 2


class AgentInfo:
    def __init__(
        self, agent_id: int, agent_device: int, ip_address: str, port: int
    ) -> None:
        self.agent_id = agent_id
        self.agent_device = agent_device
        self.ip_address = ip_address
        self.port = port
        self.status = AgentStatus.IDLE
        self.task_id = -1
        self.rank = -1

    def zmq_id(self):
        return str(self.agent_id).encode("utf-8")

    def __repr__(self) -> str:
        return f"AgentInfo[agent_id={self.agent_id}, agent_device={self.agent_device}, ip_address={self.ip_address}, status={self.status}, task_id={self.task_id}]"

    def __str__(self) -> str:
        return self.__repr__()
