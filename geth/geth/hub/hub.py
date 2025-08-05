import traceback

from loguru import logger

from geth.base.operation import AgentOperation, HubOperation, Operation, SystemOperation
from geth.base.zmq_link import ZmqServer
from geth.hub.agent_manager import AgentManager
from geth.hub.task import (
    TaskChainStatus,
    TaskScaleChain,
    TaskStatus,
    TaskTrainChain,
)
from geth.hub.task_manager import TaskManager


class GethHub:
    def __init__(self, port: int):
        self.name = "hub"
        self.zmq_server = ZmqServer(port)
        self.stop = False
        self.agent_manager = AgentManager()
        self.task_manager = TaskManager()

    def execute(self):
        while not self.stop:
            try:
                dealer_id, msg = self.zmq_server.recv()
                logger.debug(f"Hub received operation from {dealer_id}: {msg}")
                self.on_event(msg, dealer_id)
            except Exception as e:
                traceback.print_exc()
                raise e

    def on_event(self, op: Operation, dealer_id: bytes):
        if isinstance(op, SystemOperation):
            self.dispatch_system_op(op, dealer_id)
        elif isinstance(op, AgentOperation):
            self.dispatch_agent_op(op, dealer_id)
        else:
            raise Exception("Unknown operation")

    def _check_chain_and_respond(self, task_id: int):
        task_chain = self.task_manager.get_task_chain(task_id)
        if task_chain is None:
            return
        if task_chain.status == TaskChainStatus.RUNNING:
            return
        if (
            task_chain.dealer_id is not None
        ):  # todo: resp-1, change respons logic here to allow respond on any time
            success = task_chain.status == TaskChainStatus.FINISHED
            reason = f"At stage {task_chain.stage}: " + str(task_chain.failure_reason)
            self.zmq_server.send(
                task_chain.dealer_id,
                HubOperation(HubOperation.OpCode.Acknowledge, (success, reason)),
            )
        self.task_manager.pop_task_chain(task_id)

    def dispatch_agent_op(self, op: AgentOperation, dealer_id: bytes):
        # go through hub op, then go through task chain if the agent is associated with a task
        match op.op_code:
            case AgentOperation.OpCode.RegisterAgent:
                self._on_register_agent(op, dealer_id)

        agent = self.agent_manager.get_agent_by_zmq_id(dealer_id)
        if agent.task_id != -1:
            task_chain = self.task_manager.get_task_chain(agent.task_id)
            if task_chain is not None:
                task_chain.on_event(op, dealer_id)
            self._check_chain_and_respond(agent.task_id)

    def _on_register_agent(self, op: AgentOperation, dealer_id: bytes):
        agent_device, agent_ip, port = op.args
        self.agent_manager.register_agent(dealer_id, agent_device, agent_ip, port)
        self.zmq_server.send(
            dealer_id, HubOperation(HubOperation.OpCode.Acknowledge, (True, None))
        )

    def dispatch_system_op(self, op: SystemOperation, dealer_id: bytes):
        match op.op_code:
            case SystemOperation.OpCode.Timer:
                self.on_timer(op, dealer_id)
            case SystemOperation.OpCode.ListAgent:
                self.on_list_agent(op, dealer_id)
            # case SystemOperation.OpCode.ShowAgent:
            #     self._on_show_agent(op, dealer_id)
            case SystemOperation.OpCode.ListTask:
                self.on_list_task(op, dealer_id)
            case SystemOperation.OpCode.CreateTask:
                self.on_create_task(op, dealer_id)
            # case SystemOperation.OpCode.ShowTask:
            #     self._on_show_task(op, dealer_id)
            case SystemOperation.OpCode.UpdateTask:
                self.on_update_task(op, dealer_id)
            # case SystemOperation.OpCode.DeleteTask:
            #     self._on_delete_task(op, dealer_id)
            case _:
                raise Exception("Not implemented")

    def on_timer(self, op: SystemOperation, dealer_id: bytes):
        # notify every running task chain
        for task_id in self.task_manager.tasks.keys():
            task = self.task_manager.get_task(task_id)
            if task.status != TaskStatus.RUNNING:
                continue
            task_chain = self.task_manager.get_task_chain(task_id)
            assert task_chain is not None
            task_chain.on_event(op, dealer_id)

    def on_list_agent(self, op: SystemOperation, dealer_id: bytes):
        result = self.agent_manager.get_all_agents()
        self.zmq_server.send(
            dealer_id, HubOperation(HubOperation.OpCode.Acknowledge, result)
        )

    def on_list_task(self, op: SystemOperation, dealer_id: bytes):
        result = self.task_manager.get_all_tasks()
        self.zmq_server.send(
            dealer_id, HubOperation(HubOperation.OpCode.Acknowledge, result)
        )

    def on_create_task(self, op: SystemOperation, dealer_id: bytes):
        task_file = op.args[0]
        agent_ids = list(map(lambda x: int(x), op.args[1]))
        target_epoch = op.args[2] if len(op.args) > 2 else -1
        task_name = op.args[3] if len(op.args) > 3 else "task_0"
        task = self.task_manager.create_task(task_file, agent_ids, self.agent_manager, target_epoch=target_epoch, task_name=task_name)
        task_chain = TaskTrainChain(
            task,
            self.zmq_server,
            self.agent_manager,
            dealer_id,
            task_file,
            agent_ids,
        )
        task_chain.check_and_progress_pending_transaction()
        self.task_manager.push_task_chain(task.task_id, task_chain)

        self._check_chain_and_respond(task.task_id)
        # todo: resp-1, change respons logic after task chain refactor is done
        self.zmq_server.send(
            dealer_id, HubOperation(HubOperation.OpCode.Acknowledge, (True, None))
        )

    def on_update_task(self, op: SystemOperation, dealer_id: bytes):
        task_id, new_agent_ids = op.args
        new_agent_ids = list(map(lambda x: int(x), new_agent_ids))
        if not self.task_manager.has_task(task_id):
            self.zmq_server.send(
                dealer_id,
                HubOperation(HubOperation.OpCode.Acknowledge, (False, "No such task")),
            )
            return
        task = self.task_manager.get_task(task_id)
        task_chain = TaskScaleChain(
            task, self.zmq_server, self.agent_manager, dealer_id, new_agent_ids
        )
        task_chain.check_and_progress_pending_transaction()
        self.task_manager.push_task_chain(task.task_id, task_chain)

        self._check_chain_and_respond(task.task_id)
