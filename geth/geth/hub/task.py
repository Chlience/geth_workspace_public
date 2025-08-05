import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from loguru import logger

from geth.base.dist_info import DistInfo, PreScaleInfo
from geth.base.operation import (
    AgentOperation,
    HubOperation,
    Operation,
    SystemOperation,
)
from geth.base.zmq_link import ZmqServer
from geth.hub.agent_info import AgentInfo, AgentStatus
from geth.hub.agent_manager import AgentManager
from geth.trainer.elastic_ddp_trainer import (
    generate_recovery_schedule,
)
from geth.trainer.training_status import TrainingStatus


# todo: remove hack for python of lower version
def override(func):
    return func


class TaskStatus(Enum):
    WAITING = 1
    RUNNING = 2
    SCALING = 3
    FINISHED = 4


class Task:
    def __init__(
        self,
        task_id: int,
        task_file: str,
        agent_ids: list[str],
        epoch: int = -1,
        name: str = "",
        status: TaskStatus = TaskStatus.WAITING,
    ):
        self.task_id = task_id
        self.task_file = task_file
        self.agents_num = len(agent_ids)
        self.agent_ids = agent_ids
        self.status = status
        self.target_epoch = epoch
        self.name = name
        self.epoch = 0
        self.step = 0
        self.recovery_count = 0

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Task[task_name={self.name}, task_file={self.task_file}, agents_num={self.agents_num}, agent_ids={self.agent_ids}, status={self.status}, target_epoch={self.target_epoch}, epoch={self.epoch}, step={self.step}]"


# respond/chain-change/chain-terminate are all handled outside chain
class TaskChainStatus(Enum):
    RUNNING = 1
    FINISHED = 2
    FAILED = 3


def make_responder(dealer_id: bytes, zmq_server: ZmqServer):
    def respond(*args):
        zmq_server.send(
            dealer_id, HubOperation(HubOperation.OpCode.Acknowledge, tuple(args))
        )

    return respond


class GethTaskChain(ABC):
    def __init__(
        self,
        task: Task,
        zmq_server: ZmqServer,
        agent_manager: AgentManager,
        dealer_id: Optional[bytes] = None,
    ):
        self.task = task
        self.zmq_server = zmq_server
        self.agent_manager = agent_manager
        self.dealer_id = dealer_id

        self.stage = 0
        self.status = TaskChainStatus.RUNNING
        self.failure_reason = ""

    def on_event(self, op: Operation, dealer_id: bytes):
        if isinstance(op, SystemOperation):
            match op.op_code:
                case SystemOperation.OpCode.Timer:
                    pass
                case _:
                    raise NotImplementedError()
        elif isinstance(op, AgentOperation):
            match op.op_code:
                case AgentOperation.OpCode.ReportAgentStatusPeriodic:
                    self.on_report_agent_status_periodic(op, dealer_id)
                case AgentOperation.OpCode.ReportAgentWorkerInitEvent:
                    self.on_report_agent_worker_init_event(op, dealer_id)
                case AgentOperation.OpCode.ReportAgentWorkerTerminateEvent:
                    self.on_report_agent_worker_terminate_event(op, dealer_id)
                case AgentOperation.OpCode.ReportAgentTrainingPauseEvent:
                    self.on_report_agent_training_pause_event(op, dealer_id)
                case AgentOperation.OpCode.ReportAgentTrainingUnpauseEvent:
                    self.on_report_agent_training_unpause_event(op, dealer_id)
                case AgentOperation.OpCode.ReportAgentPreScaleEvent:
                    self.on_report_agent_pre_scale_event(op, dealer_id)
                case AgentOperation.OpCode.ReportAgentGetResourcesEvent:
                    self.on_report_agent_get_resources_event(op, dealer_id)
                case _:
                    raise NotImplementedError()
        else:
            raise Exception("Unknown operation")
        self.check_and_progress_pending_transaction()

    def on_report_agent_status_periodic(self, op: AgentOperation, dealer_id: bytes):
        agent_status = op.args[0]
        assert isinstance(agent_status, TrainingStatus)
        logger.debug(f"Agent {dealer_id} reported training status {op}")

        self.task.epoch = agent_status.epoch
        self.task.step = agent_status.step

    def on_report_agent_worker_init_event(self, op: AgentOperation, dealer_id: bytes):
        pass

    def on_report_agent_worker_terminate_event(
        self, op: AgentOperation, dealer_id: bytes
    ):
        pass

    def on_report_agent_training_pause_event(
        self, op: AgentOperation, dealer_id: bytes
    ):
        pass

    def on_report_agent_training_unpause_event(
        self, op: AgentOperation, dealer_id: bytes
    ):
        pass

    def on_report_agent_pre_scale_event(self, op: AgentOperation, dealer_id: bytes):
        pass

    @abstractmethod
    def check_and_progress_pending_transaction(self):
        pass

    def on_report_agent_get_resources_event(self):
        pass


class TaskTrainChain(GethTaskChain):
    """
    stage 0: on recv of CreateTask, setup data, issue startup for all new agents
    stage 1: waiting for all agent to start train
    stage 2: waiting for all old agents to finish
    """

    def __init__(
        self,
        task: Task,
        zmq_server: ZmqServer,
        agent_manager: AgentManager,
        dealer_id: bytes,
        task_file: str,
        agent_ids: list[str],
    ):
        super().__init__(task, zmq_server, agent_manager, dealer_id=dealer_id)

        self.task_file = task_file
        self.agent_ids = agent_ids

        self.agent_init_map = {}
        self.training_fini_map = {}
        self.agent_term_map = {}
        self.agents_recover_map = {}
        
        self.is_pre_scale = False

    @override
    def check_and_progress_pending_transaction(self):
        match self.stage:
            case 0:
                self._handle_stage_0()
            case 1:
                self._handle_stage_1()
            case 2:
                self._handle_stage_2()
            case 3:
                self._handle_stage_3()
            case 4:
                self._handle_stage_4()
            case 5:
                self._handle_stage_5()
            case _:
                raise Exception("Invalid stage")

    def _handle_stage_0(self):
        
        logger.debug(f"{self.task}: Created")
        
        self.is_pre_scale = (os.environ.get("GETH_ENABLE_PRESCALE", "0") == "1")
        agents = self.agent_manager.get_agents_by_ids(self.agent_ids)
        for agent in agents:
            if agent.status != AgentStatus.IDLE:
                self.status = TaskChainStatus.FAILED
                self.failure_reason = f"Agent {agent.agent_id} is not idle"
                return

        for i in range(len(agents)):
            dist_info = DistInfo(
                agents[0].ip_address,
                agents[0].port,
                i,
                len(agents),
                recovery_count=self.task.recovery_count,
                master_worker_id=agents[0].agent_id,
            )
            logger.debug(f"Init task for agent {agents[i].agent_id}")
            self.zmq_server.send(
                agents[i].zmq_id(),
                HubOperation(
                    HubOperation.OpCode.InitWorker,
                    (
                        "task_" + str(self.task.task_id),
                        self.task_file,
                        dist_info,
                        self.is_pre_scale, # 开启 PreScale 则卡住，暂时不开始训练
                        self.task.target_epoch,
                    ),
                ),
            )
            agents[i].status = AgentStatus.BUSY
            agents[i].task_id = self.task.task_id
            agents[i].rank = i
            self.agent_init_map[agents[i].agent_id] = 0
        # update task status
        self.task.status = TaskStatus.SCALING if self.is_pre_scale else TaskStatus.RUNNING
        self.task.agent_ids = [agent.agent_id for agent in agents]
        self.stage = 1

    def _handle_stage_1(self):
        # note: to handle agent change in the process
        for agent_id in self.task.agent_ids:
            if self.agent_init_map.get(agent_id, 0) == 0:
                return
        
        logger.debug(f"{self.task}: All agent inited")

        if self.is_pre_scale:
            agents = self.agent_manager.get_agents_by_ids(self.agent_ids)
            prescale_info = PreScaleInfo(
                n_agents=len(agents),
                prescale_task_id=f"task_{self.task.task_id}_init_worker",
            )
            self.zmq_server.send(
                agents[0].zmq_id(),
                HubOperation(
                    HubOperation.OpCode.PreScaleTask,
                    (prescale_info,),
                ),
            )
            self.pre_scale_state = 0
            self.stage = 2
        else:
            self.stage = 4

    def _handle_stage_2(self):
        if (self.is_pre_scale and self.pre_scale_state != 1):
            return
        agents = self.agent_manager.get_agents_by_ids(self.agent_ids)
        for i in range(len(agents)):
            dist_info = dist_info = DistInfo(
                agents[0].ip_address,
                agents[0].port,
                i,
                len(agents),
                recovery_count=self.task.recovery_count,
                master_worker_id=agents[0].agent_id,
            )
            self.zmq_server.send(
                agents[i].zmq_id(),
                HubOperation(
                    HubOperation.OpCode.UnpauseTask,
                    (
                        False,      # rec_flag
                        None,       # recovery_schedule
                        dist_info,  # 
                        True,
                    ),
                ),
            )
            self.agents_recover_map[agents[i].agent_id] = 0
        
        self.stage = 3
        
    def _handle_stage_3(self):
        if not all([stat == 1 for stat in self.agents_recover_map.values()]):
            return
        
        logger.debug(f"{self.task}: All agent recoveried")
        
        self.task.status = TaskStatus.RUNNING
        self.stage = 4

    def _handle_stage_4(self):
        # note: to handle agent change in the process
        for agent_id in self.task.agent_ids:
            if self.training_fini_map.get(agent_id, 0) == 0:
                return
        
        logger.debug(f"{self.task}: Finished")
        
        self.training_fini_map = {}

        for agent_id in self.task.agent_ids:
            agent = self.agent_manager.get_agent(agent_id)
            self.zmq_server.send(
                agent.zmq_id(),
                HubOperation(HubOperation.OpCode.TerminateWorker, ()),
            )
            self.agent_term_map[agent_id] = 0
        self.stage = 5

    def _handle_stage_5(self):
        for agent_id in self.task.agent_ids:
            if self.agent_term_map.get(agent_id, 0) == 0:
                return
        logger.debug(f"{self.task}: All agent terminated")
        self.status = TaskChainStatus.FINISHED
        for agent_id in self.task.agent_ids:
            agent = self.agent_manager.get_agent(agent_id)
            agent.status = AgentStatus.IDLE
            agent.task_id = -1
            agent.rank = -1
        self.task.status = TaskStatus.FINISHED
        self.task.agent_ids = []
        logger.info(f"{self.task}: Hub Exit")
        exit(0)

    @override
    def on_report_agent_worker_init_event(self, op: AgentOperation, dealer_id: bytes):
        agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
        self.agent_init_map[agent.agent_id] = 1

    @override
    def on_report_agent_worker_terminate_event(
        self, op: AgentOperation, dealer_id: bytes
    ):
        agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
        self.agent_term_map[agent.agent_id] = 1

    @override
    def on_report_agent_status_periodic(self, op: AgentOperation, dealer_id: bytes):
        super().on_report_agent_status_periodic(op, dealer_id)

        if op.args[0].running is False:
            agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
            self.training_fini_map[agent.agent_id] = 1

    @override
    def on_report_agent_pre_scale_event(self, op: AgentOperation, dealer_id: bytes):
        self.pre_scale_state = 1
        logger.debug(f"{self.task}: Agent {dealer_id} reported prescale finished")

    @override
    def on_report_agent_training_unpause_event(
        self, op: AgentOperation, dealer_id: bytes
    ):
        agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
        self.agents_recover_map[agent.agent_id] = 1


class TaskScaleChain(GethTaskChain):
    """
    stage 0: on recv of UpdateTask, setup data, issue startup for all new agents
    # todo: latest: send prescale to agent 0
    stage 1: waiting for new agents to startup
        -> terminate all if training finished in the process
        -> start prepartition
        -> to stage 2 if all new agents started, issue pause to all old agents
    stage 2: waiting for all old agents to pause
        -> terminate all if training finished in the process
        -> to stage 3 if all new agents started, issue recovery for all agents
    stage 3: waiting for all agents to recover
        -> terminate when all agents reported recovery, finishes transaction
    """

    def __init__(
        self,
        task: Task,
        zmq_server: ZmqServer,
        agent_manager: AgentManager,
        dealer_id: bytes,
        new_agent_ids: list[str],
    ):
        super().__init__(task, zmq_server, agent_manager, dealer_id=dealer_id)

        self.new_agent_ids = new_agent_ids

        self.old_agents: List[AgentInfo] = []
        self.new_agents: List[AgentInfo] = []
        self.recovery_schedule = {}

        self.new_agent_init_map = {}
        self.pre_scale_state = -1
        self.old_agent_pause_map = {}
        self.agents_recover_map = {}
        self.old_agents_fini_map = {}
        self.agents_get_resources_map = {}

    @override
    def check_and_progress_pending_transaction(self):
        match self.stage:
            case 0:
                self._handle_stage_0()
            case 1:
                self._handle_stage_1()
            case 2:
                self._handle_stage_2()
            case 3:
                self._handle_stage_3()
            case 4:
                self._handle_stage_4()
            case _:
                raise Exception("Invalid stage")

    def _handle_stage_0(self):
        logger.debug(f"{self.task}: Scaling with new agents {self.new_agent_ids}")
        
        if self.task is None:
            self.status = TaskChainStatus.FAILED
            self.failure_reason = "Task not found"
            return
        if self.task.status is not TaskStatus.RUNNING:
            self.status = TaskChainStatus.FAILED
            self.failure_reason = "Task not running"
            return
        if self.new_agent_ids == self.task.agent_ids:
            self.status = TaskChainStatus.FINISHED
            return

        self.task.status = TaskStatus.SCALING
        self.task.recovery_count += 1
        # 1. determine new agent ids
        old_agent_ids = self.task.agent_ids.copy()
        new_agent_ids = self.new_agent_ids

        # 1.5 check agents
        # todo: add extra_task_info field
        # assert old_agent_ids[0] == new_agent_ids[0]
        # 在 generate_recovery_schedule 中会检查

        # 2. generate recovery schedule
        self.old_agents = [
            self.agent_manager.get_agent(agent_id) for agent_id in old_agent_ids
        ]
        self.new_agents = [
            self.agent_manager.get_agent(agent_id) for agent_id in new_agent_ids
        ]
        self.recovery_schedule = generate_recovery_schedule(
            self.old_agents, self.new_agents, self.task.recovery_count
        )

        self.task.status = TaskStatus.SCALING
        self.task.agents_num = len(self.new_agents)
        for new_agent in self.new_agents:
            new_agent.status = AgentStatus.BUSY
            new_agent.task_id = self.task.task_id

        for agent in self.new_agents:
            if agent in self.old_agents:
                continue
            self.zmq_server.send(
                agent.zmq_id(),
                HubOperation(
                    HubOperation.OpCode.InitWorker,
                    (
                        f"task_{self.task.task_id}",
                        self.task.task_file,
                        DistInfo(
                            agent.ip_address,
                            agent.port,
                            -1,
                            0,
                            recovery_count=self.task.recovery_count,
                            master_worker_id=self.old_agents[0].agent_id,
                        ),
                        True,
                        self.task.target_epoch,
                    ),
                ),
            )
            self.new_agent_init_map[agent.agent_id] = 0
            agent.status = AgentStatus.BUSY
            agent.task_id = self.task.task_id

        # 3. send prescale to agent 0
        if os.environ.get("GETH_ENABLE_PRESCALE", "0") == "1":
            prescale_info = PreScaleInfo(
                n_agents=len(self.new_agents),
                prescale_task_id=f"task_{self.task.task_id}_{self.task.recovery_count}",
            )
            self.zmq_server.send(
                self.new_agents[0].zmq_id(),
                HubOperation(
                    HubOperation.OpCode.PreScaleTask,
                    (prescale_info,),
                ),
            )
            self.pre_scale_state = 0

        self.stage = 1

    def _handle_stage_1(self):
        if not all([stat == 1 for stat in self.new_agent_init_map.values()]):
            return
        if (
            os.environ.get("GETH_ENABLE_PRESCALE", "0") == "1"
            and self.pre_scale_state != 1
        ):
            return
        
        # 先让新 agent 获得资源，然后再 pause old agent
        for agent in self.new_agents:
            if agent not in self.old_agents:
                self.zmq_server.send(
                    agent.zmq_id(),
                    HubOperation(HubOperation.OpCode.GetResources, ()),
                )
                self.agents_get_resources_map[agent.agent_id] = 0
            
        self.stage = 2
        
    def _handle_stage_2(self):
        if not all([stat == 1 for stat in self.agents_get_resources_map.values()]):
            return
        
        logger.debug(f"{self.task}: All agents get resources for scale")

        # pause all old agents
        for agent in self.old_agents:
            self.zmq_server.send(
                agent.zmq_id(),
                HubOperation(HubOperation.OpCode.PauseTask, ()),
            )
            self.old_agent_pause_map[agent.agent_id] = 0
        
        self.stage = 3

    def _handle_stage_3(self):
        if not all([stat == 1 for stat in self.old_agent_pause_map.values()]):
            return

        logger.debug(f"{self.task}: All old agents paused")

        # recover all agents
        new_agent_ids = [agent.agent_id for agent in self.new_agents]
        for agent_id, recovery_sch in self.recovery_schedule.items():
            agent = self.agent_manager.get_agent(agent_id)
            assert agent is not None

            is_new_agent = agent_id in new_agent_ids
            if is_new_agent:
                rank = new_agent_ids.index(agent_id)
                dist_info = DistInfo(
                    self.new_agents[0].ip_address,
                    self.new_agents[0].port,
                    rank,
                    len(new_agent_ids),
                    recovery_count=self.task.recovery_count,
                )
                agent.status = AgentStatus.BUSY
                agent.task_id = self.task.task_id
                agent.rank = rank
                self.agents_recover_map[agent_id] = 0
            else:
                rank = -1
                dist_info = DistInfo(
                    agent.ip_address,
                    agent.port,
                    rank,
                    0,
                    recovery_count=self.task.recovery_count,
                )
                self.agents_recover_map[agent_id] = 0
                self.old_agents_fini_map[agent_id] = 0
            # 老agent如果退出会导致新agent没法恢复,所以还是要先让他们都启动
            # 由于已经提前获取资源，无需等待
            need_wait = False
            # need_wait = is_new_agent and (agent not in self.old_agents)
            self.zmq_server.send(
                agent.zmq_id(),
                HubOperation(
                    HubOperation.OpCode.UnpauseTask,
                    (
                        True,
                        recovery_sch,
                        dist_info,
                        need_wait,
                    ),
                ),
            )
        self.stage = 4

    def _handle_stage_4(self):
        if not all([stat == 1 for stat in self.agents_recover_map.values()]):
            return
        if not all([stat == 1 for stat in self.old_agents_fini_map.values()]):
            return
        
        logger.debug(f"{self.task}: All agents recovered for scale")
        
        self.task.status = TaskStatus.RUNNING
        self.task.agent_ids = [agent.agent_id for agent in self.new_agents]
        self.task.agents_num = len(self.task.agent_ids)
        for agent in self.old_agents:
            if agent in self.new_agents:
                continue
            agent.status = AgentStatus.IDLE
            agent.task_id = -1
            agent.rank = -1
        self.status = TaskChainStatus.FINISHED

    @override
    def on_report_agent_training_pause_event(
        self, op: AgentOperation, dealer_id: bytes
    ):
        agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
        self.old_agent_pause_map[agent.agent_id] = 1

    @override
    def on_report_agent_training_unpause_event(
        self, op: AgentOperation, dealer_id: bytes
    ):
        agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
        # ! for any agents since we are not sending scale done signal
        self.agents_recover_map[agent.agent_id] = 1

    @override
    def on_report_agent_worker_init_event(self, op: AgentOperation, dealer_id: bytes):
        agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
        self.new_agent_init_map[agent.agent_id] = 1

    @override
    def on_report_agent_worker_terminate_event(
        self, op: AgentOperation, dealer_id: bytes
    ):
        agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
        assert agent in self.old_agents
        self.old_agents_fini_map[agent.agent_id] = 1  # for old agents

    @override
    def on_report_agent_status_periodic(self, op: AgentOperation, dealer_id: bytes):
        """
        对于的Agent报告训练结束的情况，这只能是第三步中old agent知道训练结束了，主动报告自身训练结束
        因此，需要向old agent发送terminate task的指令，等待worker终结后在on_report_agent_worker_terminate_event更新其状态
        """
        super().on_report_agent_status_periodic(op, dealer_id)

        if op.args[0].running is False:
            agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
            logger.debug(f"Agent {dealer_id} reported training finished")
            assert self.stage == 3
            assert agent in self.old_agents
            self.zmq_server.send(
                agent.zmq_id(),
                HubOperation(HubOperation.OpCode.TerminateWorker, ()),
            )

    @override
    def on_report_agent_pre_scale_event(self, op: AgentOperation, dealer_id: bytes):
        self.pre_scale_state = 1
        logger.debug(f"Agent {dealer_id} reported prescale finished")

    @override
    def on_report_agent_get_resources_event(
        self, op: AgentOperation, dealer_id: bytes
    ):
        agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
        self.agents_get_resources_map[agent.agent_id] = 1