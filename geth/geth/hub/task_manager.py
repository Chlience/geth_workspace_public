from typing import Dict, List, Optional

from loguru import logger

from geth.hub.agent_manager import AgentManager
from geth.hub.task import GethTaskChain, Task, TaskStatus


class TaskManager:
    def __init__(self):
        self.tasks: Dict[int, Task] = {}
        self.task_chains: Dict[int, List[GethTaskChain]] = {}
        self.counter = 0

    def create_task(
        self, task_file: str, agent_ids: list[str], agent_manager: AgentManager, target_epoch: int = -1, task_name: str = ""
    ) -> Task:
        task = Task(self.counter, task_file, agent_ids, target_epoch, task_name)
        task.status = TaskStatus.WAITING
        self.tasks[self.counter] = task
        self.counter += 1
        logger.info(f"Created task: {task}")
        return task

    def push_task_chain(self, task_id: int, task_chain: GethTaskChain):
        if task_id not in self.task_chains:
            self.task_chains[task_id] = []
        self.task_chains[task_id].append(task_chain)

    def pop_task_chain(self, task_id: int):
        assert task_id in self.task_chains
        self.task_chains[task_id].pop()

    def get_task_chain(self, task_id: int) -> Optional[GethTaskChain]:
        if task_id not in self.task_chains:
            return None
        if len(self.task_chains[task_id]) == 0:
            return None
        return self.task_chains[task_id][-1]

    def get_all_tasks(self) -> List[Task]:
        return list(self.tasks.values())

    def get_task(self, task_id: int) -> Task:
        return self.tasks[task_id]

    def has_task(self, task_id: int) -> bool:
        return task_id in self.tasks
