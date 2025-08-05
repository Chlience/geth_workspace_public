from geth.hub.agent_info import AgentStatus
from geth.hub.agent_manager import AgentManager


def test_register_agent():
    agent_manager = AgentManager()
    agent_id = b"agent1"
    agent_device = 1
    ip_address = "192.168.0.1"
    port = 32001

    agent_manager.register_agent(agent_id, agent_device, ip_address, port)

    # Assert that the agent is registered correctly
    assert agent_manager.get_idle_agents(1)[0].agent_id == agent_id
    assert agent_manager.get_idle_agents(1)[0].agent_device == agent_device
    assert agent_manager.get_idle_agents(1)[0].ip_address == ip_address
    assert agent_manager.get_idle_agents(1)[0].port == port


def test_get_idle_agents():
    agent_manager = AgentManager()

    agent_id_1 = b"agent1"
    agent_device_1 = 1
    ip_address_1 = "192.168.0.1"
    port1 = 32001
    agent_manager.register_agent(agent_id_1, agent_device_1, ip_address_1, port1)

    agent_id_2 = b"agent2"
    agent_device_2 = 2
    ip_address_2 = "192.168.0.2"
    port2 = 32002
    agent_manager.register_agent(agent_id_2, agent_device_2, ip_address_2, port2)
    agent_manager.agents[agent_id_2].status = AgentStatus.BUSY

    agent_id_3 = b"agent3"
    agent_device_3 = 3
    ip_address_3 = "192.168.0.3"
    port3 = 32003
    agent_manager.register_agent(agent_id_3, agent_device_3, ip_address_3, port3)

    # Assert that the registered agent is returned as idle
    idle_agents = agent_manager.get_idle_agents(2)
    assert len(idle_agents) == 2
    assert idle_agents[0].agent_id == agent_id_1
    assert idle_agents[0].agent_device == agent_device_1
    assert idle_agents[0].ip_address == ip_address_1
    assert idle_agents[0].port == port1

    assert idle_agents[1].agent_id == agent_id_3
    assert idle_agents[1].agent_device == agent_device_3
    assert idle_agents[1].ip_address == ip_address_3
    assert idle_agents[1].port == port3


def test_get_idle_agents_with_count_zero():
    agent_manager = AgentManager()
    agent_id = b"agent1"
    agent_device = 1
    ip_address = "192.168.0.1"
    port = 32001

    agent_manager.register_agent(agent_id, agent_device, ip_address, port)

    # Assert that an empty list is returned when count is zero
    idle_agents = agent_manager.get_idle_agents(0)
    assert len(idle_agents) == 0


def test_get_idle_agents_with_insufficient_agents():
    agent_manager = AgentManager()
    agent_id = b"agent1"
    agent_device = 1
    ip_address = "192.168.0.1"
    port = 32001

    agent_manager.register_agent(agent_id, agent_device, ip_address, port)

    # Assert that an empty list is returned when there are insufficient idle agents
    idle_agents = agent_manager.get_idle_agents(2)
    assert len(idle_agents) == 0
