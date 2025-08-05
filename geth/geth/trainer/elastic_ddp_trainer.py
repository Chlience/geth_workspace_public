import json
from enum import Enum
from typing import Dict, List

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel

from geth.base.dist_info import DistInfo
from geth.base.operation import DaemonOperation
from geth.data.comm_manager import CommManager, TopoManager
from geth.data.dataloader import ElasticDataLoader
from geth.data.elas_data import (
    ElasData,
    ElasDataManager,
    ElasDataType,
    ElasDistribScheme,
)
from geth.data.sampler import ElasticDistributedSampler
from geth.hub.agent_info import AgentInfo
from geth.trainer.base_trainer import GethBaseTrainer, GethRecoveryShedule
from geth.trainer.training_status import TrainerStopReason


class DDPRecoveryType(Enum):
    DEPREACATED_AGENTS = 0
    NEW_AGENTS = 1
    COMMON_AGENTS = 2


class GethDDPRecoveryShedule(GethRecoveryShedule):
    def __init__(
        self,
        recovery_dist_info: DistInfo,
        new_worker_list: List[int],
        rank_map: Dict[int, int],
        setup_rank_map: Dict[int, Dict[str, int]],
        agent_type: DDPRecoveryType = DDPRecoveryType.COMMON_AGENTS,
        should_save_extra_data: bool = False,
    ):
        super().__init__()
        self.recovery_dist_info = recovery_dist_info
        self.new_worker_list = new_worker_list
        self.rank_map = rank_map
        self.setup_rank_map = setup_rank_map
        self.agent_type = agent_type
        self.should_save_extra_data = should_save_extra_data

    def get_new_distib_scheme(self, data: ElasData):
        return ElasDistribScheme(None, 1, {0: self.new_worker_list})

    def get_new_distib_scheme_tp(self, data: ElasData):
        return ElasDistribScheme(
            [0],
            len(self.new_worker_list),
            {i: [i] for i in range(len(self.new_worker_list))},
        )

    def get_old_distib_scheme_tp(self, data: ElasData):
        old_cnt = 0
        for r in self.setup_rank_map:
            if self.setup_rank_map[r]["old"] != -1:
                old_cnt += 1
        return ElasDistribScheme([0], old_cnt, {i: [i] for i in range(old_cnt)})

    def __repr__(self):
        return f"GethDDPRecoveryShedule(recovery_dist_info={self.recovery_dist_info}, new_worker_list={self.new_worker_list}, rank_map={self.rank_map}, setup_rank_map={self.setup_rank_map}, agent_type={self.agent_type}, should_save_extra_data={self.should_save_extra_data})"


def generate_recovery_schedule(
    old_agents: List[AgentInfo], new_agents: List[AgentInfo], task_recovery_count
) -> Dict[int, GethDDPRecoveryShedule]:
    recovery_schedule = {}

    # 1.find all gents that are (1) in both new and old (2) only in new (3) only in old
    common_agents = [agent for agent in old_agents if agent in new_agents]
    new_only_agents = [agent for agent in new_agents if agent not in old_agents]
    old_only_agents = [agent for agent in old_agents if agent not in new_agents]
    all_agents = common_agents + new_only_agents + old_only_agents
    new_agent_ids = [agent.agent_id for agent in new_agents]

    assert common_agents[0].rank == 0, (
        "1st common agents should have rank 0 due to GNN trainer design"
    )

    has_saver = False

    rank_map = {}
    setup_rank_map = {}
    for idx, agent in enumerate(all_agents):
        rank_map[agent.agent_id] = idx
        setup_rank_map[idx] = {"old": -1, "new": -1}
        setup_rank_map[idx]["new"] = (
            new_agents.index(agent) if agent in new_agents else -1
        )
        setup_rank_map[idx]["old"] = (
            old_agents.index(agent) if agent in old_agents else -1
        )

    for idx, agent in enumerate(all_agents):
        dist_info = DistInfo(
            all_agents[0].ip_address,
            all_agents[0].port + 1,  # use port + 1 for recover dist group
            idx,
            len(all_agents),
            recovery_count=task_recovery_count,
        )

        agent_type = None
        if agent in new_only_agents:
            agent_type = DDPRecoveryType.NEW_AGENTS
        elif agent in common_agents:
            agent_type = DDPRecoveryType.COMMON_AGENTS
        else:
            agent_type = DDPRecoveryType.DEPREACATED_AGENTS

        should_save_extra_data = False
        if agent_type != DDPRecoveryType.NEW_AGENTS and not has_saver:
            should_save_extra_data = True
            has_saver = True

        recovery_schedule[agent.agent_id] = GethDDPRecoveryShedule(
            recovery_dist_info=dist_info,
            new_worker_list=new_agent_ids,
            rank_map=rank_map,
            setup_rank_map=setup_rank_map,
            agent_type=agent_type,
            should_save_extra_data=should_save_extra_data,
        )

    return recovery_schedule


class GethElasticDDPTrainer(GethBaseTrainer):
    def setup_dist(self, dist_info):
        dist.init_process_group(
            init_method=dist_info.get_init_url(),
            rank=dist_info.rank,
            world_size=dist_info.world_size,
        )
        dist.barrier()
        logger.debug(f"Rank {dist_info.rank} setup dist done")
        self.model = DistributedDataParallel(self.model)

    def sync_running_status(self):
        # note: possible race condition if we just don't sync stats here
        running_value = 1 if self.running else 0
        running_tensor = torch.tensor([running_value])
        dist.all_reduce(running_tensor, op=ReduceOp.SUM)

        if (
            running_tensor[0] != 0
            and running_tensor[0] != dist.get_world_size()
            and self.running
        ):
            logger.debug("Inconsisent running state, waiting!")
            self.handle_msg([DaemonOperation.OpCode.StopTraining])

        dist.barrier()

    def teardown_dist(self):
        if isinstance(self.model, DistributedDataParallel):
            self.model = self.model.module
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def setup_dataloader(self, task_config, dist_info, extra_info=None):
        dataset = task_config.get_dataset()
        cfg = task_config.get_cfg()["dataloader"]
        sampler = ElasticDistributedSampler(
            dataset,  # type: ignore[arg-type]
            num_replicas=dist_info.world_size if dist_info.world_size != 0 else 1,
            rank=dist_info.rank if dist_info.rank != -1 else 0,
            shuffle=cfg["random"],
        )
        self.dataloader = ElasticDataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            shuffle=False,
        )

    def stage_local_training_resource(self):
        dataloader_state = self.dataloader.save_dataloader_state()
        training_state = {
            "dataloader": dataloader_state,
            "metadata": {
                "epoch": self.epoch,
                "step": self.step,
            },
        }
        self.recover_data = training_state

    def recover_training(self, recovery_schedule):
        assert isinstance(recovery_schedule, GethDDPRecoveryShedule)

        self.model.cuda()
        # check: maybe need to move optimizer to cuda

        distributed_store = dist.TCPStore(  # pyright: ignore
            recovery_schedule.recovery_dist_info.master_addr,
            recovery_schedule.recovery_dist_info.master_port,
            recovery_schedule.recovery_dist_info.world_size,
            is_master=recovery_schedule.recovery_dist_info.rank == 0,
        )
        dist.init_process_group(
            store=distributed_store,
            rank=recovery_schedule.recovery_dist_info.rank,
            world_size=recovery_schedule.recovery_dist_info.world_size,
        )
        comm_group = dist.new_group()
        tensor = torch.ones([1], dtype=torch.int32, device=torch.device("cuda"))
        dist.all_reduce(tensor, op=ReduceOp.SUM, group=comm_group)

        dist.barrier()
        logger.debug("Training recovery group setup done")

        # build topo and comm manager
        topo_manager = TopoManager(self.worker_id, recovery_schedule.rank_map)
        comm_manager = CommManager(topo_manager, comm_group)

        # issue transfer
        self.apply_new_elastic_schedule(
            self.elas_data_manager, recovery_schedule, comm_manager
        )

        # selected new ones sets recovery data
        if recovery_schedule.should_save_extra_data:
            distributed_store.set("recovery_data", json.dumps(self.recover_data))

        comm_manager.commit()
        self.apply_elastic_data()

        dist.barrier()

        if recovery_schedule.agent_type == DDPRecoveryType.NEW_AGENTS:
            self.recover_data = json.loads(distributed_store.get("recovery_data"))

        dist.barrier()

        dist.destroy_process_group()
        del distributed_store
        logger.debug("Finished transfer of all recovery data")

        # recover from state
        assert self.recover_data is not None
        self.dataloader.load_dataloader_state(self.recover_data["dataloader"])
        # 4. recover epoch and more
        self.epoch = self.recover_data["metadata"]["epoch"]
        self.step = self.recover_data["metadata"]["step"]

        self.recover_data = None

        # stop if it is an deprecated agent
        if recovery_schedule.agent_type == DDPRecoveryType.DEPREACATED_AGENTS:
            self.running = False
            self.stop_reason = TrainerStopReason.STOPPED_BY_AGENT

    def setup_elastic_data(self):
        for idx, param in enumerate(self.model.parameters()):
            assert isinstance(param, torch.Tensor)
            name = f"param_{idx}"
            data = param
            distrib_scheme = ElasDistribScheme(None, 1, {0: [0]})
            self.elas_data_manager.register_tensor_data(
                distrib_scheme, ElasDataType.MODEL_PARAM, data, name
            )

        for gid, group in enumerate(self.optimizer.param_groups):
            for pid, p in enumerate(group["params"]):
                if isinstance(p, torch.Tensor):
                    state = self.optimizer.state[p]
                    keys_sorted = sorted(state.keys())
                    for k in keys_sorted:
                        name = f"optim_{gid}_{pid}_{k}"
                        data = state[k]
                        distrib_scheme = ElasDistribScheme(None, 1, {0: [0]})
                        self.elas_data_manager.register_tensor_data(
                            distrib_scheme, ElasDataType.OPTIMIZER_STATE, data, name
                        )
                else:
                    logger.error("Unknown param:", p)

        # self.elas_data_manager.summary()

    def apply_elastic_data(self):
        pass

    def apply_new_elastic_schedule(
        self,
        elas_data_manager: ElasDataManager,
        recovery_schedule: GethDDPRecoveryShedule,
        comm_manager: CommManager,
    ):
        for name, data in elas_data_manager.data_dict.items():
            new_scheme = recovery_schedule.get_new_distib_scheme(data)
            assert new_scheme is not None, f"Elastic data {name} has no new scheme!"
            data.change_distib_scheme(new_scheme, comm_manager)
