class DistInfo:
    def __init__(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        recovery_count: int = -1,
        master_worker_id: int = -1,
    ) -> None:
        self.master_addr = master_addr
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.recovery_count = recovery_count
        self.master_worker_id = master_worker_id

    def get_init_url(self):
        return f"tcp://{self.master_addr}:{self.master_port}"

    def __repr__(self) -> str:
        return f"DistInfo(master_addr={self.master_addr}, master_port={self.master_port}, rank={self.rank}, world_size={self.world_size}, master_worker_id={self.master_worker_id})"


class PreScaleInfo:
    def __init__(self, n_agents: int, prescale_task_id: str) -> None:
        self.n_agents = n_agents
        self.prescale_task_id = prescale_task_id

    def __repr__(self) -> str:
        return f"PreScaleInfo(n_agents={self.n_agents}, prescale_task_id={self.prescale_task_id})"
