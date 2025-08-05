from typing import List

from loguru import logger

from geth.hub.agent_info import AgentInfo, AgentStatus


class AgentManager:
    def __init__(self) -> None:
        self.agents = {}

    def register_agent(
        self, agent_id_bytes: bytes, agent_device: int, ip_address: str, port: int
    ) -> None:
        agent_id = int(agent_id_bytes.decode("utf-8"))
        assert agent_id not in self.agents
        agent_state = AgentInfo(agent_id, agent_device, ip_address, port)
        self.agents[agent_id] = agent_state
        logger.info(f"Registered agent {agent_state}")

    def get_idle_agents(self, count: int) -> List[AgentInfo]:
        # ? possible improvement: topology-aware agent selection
        agent_list = [
            agent for agent in self.agents.values() if agent.status == AgentStatus.IDLE
        ]
        if len(agent_list) >= count:
            agent_list = agent_list[:count]
        else:
            logger.debug(
                f"Requested {count} agents, but only {len(agent_list)} available"
            )
            return []
        logger.debug(f"Get {count} idle agents")
        return agent_list

    def get_idle_agents_cnt(self, count: int) -> int:
        agent_list = [
            agent for agent in self.agents.values() if agent.status == AgentStatus.IDLE
        ]
        return len(agent_list)

    def get_all_agents(self) -> List[AgentInfo]:
        agent_list = [agent for agent in self.agents.values()]
        return agent_list

    def get_agent(self, agent_id: int) -> AgentInfo:
        return self.agents[agent_id]

    def get_agents_by_ids(self, agent_ids: List[int]) -> List[AgentInfo]:
        return [self.agents[agent_id] for agent_id in agent_ids]

    def get_agent_by_zmq_id(self, agent_id_bytes: bytes) -> AgentInfo:
        agent_id = int(agent_id_bytes.decode("utf-8"))
        return self.agents[agent_id]
