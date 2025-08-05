import copy
import json
import os
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ragdoll
import scipy
import torch
import torch.distributed as dist
import torch.utils.data
from dgl import DGLGraph
from loguru import logger
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel

from geth.base.dist_info import DistInfo
from geth.base.operation import DaemonOperation, TrainerOperation
from geth.data.comm_manager import CommManager, TopoManager
from geth.data.dataloader import ElasticDataLoader
from geth.data.dataset_adapter import GethDatasetAdapter
from geth.data.elas_data import (
    ElasDataManager,
    ElasDataType,
    ElasDistribScheme,
)
from geth.data.graph_dataset import get_gnn_dataset
from geth.data.sampler import ElasticSequentialSampler
from geth.trainer.base_trainer import GethBaseTrainer
from geth.trainer.elastic_ddp_trainer import DDPRecoveryType, GethDDPRecoveryShedule
from geth.trainer.training_status import TrainerStopReason

Args = namedtuple("Args", ["dataset", "feat_size"])


class SingleItemDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # Initialize with any necessary data, if needed
        self.data = data  # Example data, you can put any single data point here

    def __len__(self):
        # Return the length of the dataset
        return 1

    def __getitem__(self, idx):
        # Return the data point at index `idx`
        if idx >= len(self):
            raise IndexError("Index out of range")
        return self.data


def pad_to_128(a):
    assert len(a.shape) == 2
    d = a.shape[1]
    padding = (128 - d % 128) % 128
    if padding == 0:
        return a
    return np.pad(a, ((0, 0), (0, padding)), constant_values=0)


def pad_to_128_torch(a):
    assert len(a.shape) == 2
    d = a.shape[1]
    padding = (128 - d % 128) % 128
    if padding == 0:
        return a
    return torch.nn.functional.pad(a, (0, padding), mode="constant", value=0)


class GethElasticDDPGNNTrainer(GethBaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.full_n_nodes = 0
        self.full_graph = scipy.sparse.csr_matrix((0, 0))
        self.dgcl_inited = False

        self.prev_gid2pid = None
        self.prev_num_local_nodes = None
        self.prev_gid2lid = None

        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)
        self.prepart_future = None
        self.distributed_store = None

    def setup_dgcl_dist(self, dist_info: DistInfo):
        os.environ["RAGDOLL_USE_MPI"] = "0"
        os.environ["RAGDOLL_MASTER_ADDR"] = str(dist_info.master_addr)
        os.environ["RAGDOLL_PORT"] = str(dist_info.master_port + 2)
        os.environ["RAGDOLL_RANK"] = str(dist_info.rank)
        os.environ["RAGDOLL_WORLD_SIZE"] = str(dist_info.world_size)

        logs_name = "GCCL.RANK." + str(dist_info.rank)
        ragdoll.init_logs(logs_name)

        # ! ragdoll need to know all devices so can't use CUDA_VISIBLE_DEVICES to control devices
        device_id = int(os.environ["GETH_DEVICE"])
        ragdoll.init(device_id=device_id)

    def setup_dist(self, dist_info):
        if self.distributed_store is None:
            self.distributed_store = dist.TCPStore(  # pyright: ignore
                dist_info.master_addr,
                dist_info.master_port,
                is_master=dist_info.rank == 0,
            )
            logger.debug(
                f"[{dist_info.rank}] Dist store initialized, master addr: {dist_info.master_addr}, port: {dist_info.master_port}"
            )
        train_store = dist.PrefixStore(
            f"train_{dist_info.recovery_count}_", self.distributed_store
        )
        dist.init_process_group(
            store=train_store,
            rank=dist_info.rank,
            world_size=dist_info.world_size,
        )
        ragdoll.set_comm_pattern(self.cfg["comm_pattern"])
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
        if self.dgcl_inited:
            ragdoll.deinit()
            ragdoll.deinit_logs()
            self.dgcl_inited = False
        if isinstance(self.model, DistributedDataParallel):
            self.model = self.model.module
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def setup_dataloader(self, task_config, dist_info, extra_info=None):
        # features = torch.FloatTensor(extra_info["features"]).cuda()
        # labels = torch.LongTensor(extra_info["labels"]).cuda()
        features = extra_info["features"]
        labels = extra_info["labels"]
        self.dataset = GethDatasetAdapter(SingleItemDataset((features, labels)))

        def cust_collate(batch_dict):
            return batch_dict[0]

        self.dataloader = ElasticDataLoader(
            self.dataset,
            batch_size=1,
            collate_fn=cust_collate,
            sampler=ElasticSequentialSampler(self.dataset),
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

    # todo: unify with base method
    def recover_training(self, recovery_schedule, distributed_store, comm_group):
        logger.debug("Start recovery of DDP Part")

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

        logger.debug("Finished transfer of DDP recovery data")

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

    def init_worker(self, rank=-1, world_size=1, master_device_id=0, target_epoch=-1):
        device_id = torch.cuda.current_device()
        self.cfg = self.task_config.get_cfg()
        # 初始训练
        # 1. hub启动所有进程，0号进程的init读取dgl图，其他进程不动
        self.local_features = torch.empty(
            [0, self.cfg["dataloader"]["feat_size"]], device="cuda"
        )
        self.local_labels = torch.empty([0], dtype=torch.int32, device="cuda")
        self.local_train_mask = torch.empty([0], dtype=torch.int32, device="cuda")
        if rank == 0:
            logger.debug("Start loading full graph data")

            (
                dgl_graph,
                num_classed,
                self.full_labels,
                self.full_train_mask,
                self.mini_graph_info,
            ) = get_gnn_dataset(self.cfg["dataloader"]["dataset"])
            self.full_n_nodes = dgl_graph.number_of_nodes()
            self.full_features = dgl_graph.ndata["feat"]
            self.full_graph = dgl_graph.adj_external(scipy_fmt="csr")
            self.full_labels = self.full_labels.to(torch.int32)
            self.full_train_mask = self.full_train_mask.to(torch.int32)

            self.local_features = self.full_features
            self.local_labels = self.full_labels
            self.local_train_mask = self.full_train_mask

            self.prev_gid2pid = torch.zeros([self.full_n_nodes], dtype=torch.int32)
            self.prev_num_local_nodes = torch.tensor(
                [self.full_n_nodes], dtype=torch.int32
            )
            self.prev_gid2lid = torch.arange(self.full_n_nodes, dtype=torch.int32)

        self.model = self.task_config.get_model()
        self.optimizer = self.task_config.get_optimizer()
        self.cfg = self.task_config.get_cfg()
        # later: 可以分featsize读取存储数据，然后再分发等等
        self.epoch = 0
        self.step = 0
        self.target_epochs = target_epoch if target_epoch > 0 else self.cfg["target_epochs"]
        self.init_optimizer_state(self.optimizer)
        self.setup_elastic_data(master_device_id)

    def _partation_graph_and_distribute(self, dist_info):
        # ! 这里默认了0号进程始终有对应的数据
        if dist_info.rank == 0:
            logger.debug("Try to partition")
            # sg_n, sg_xadj, sg_adjncy = ragdoll.partition_graph(
            #     self.full_n_nodes, self.full_graph.indptr, self.full_graph.indices, self.mini_graph_info
            # )
            # prepart_info = ragdoll.pre_partition_graph(
            #     dist_info.world_size,
            #     self.full_n_nodes,
            #     self.full_graph.indptr,
            #     self.full_graph.indices,
            #     self.mini_graph_info,
            # )
            start = time.time()
            print("dist_info.world_size", dist_info.world_size)
            if self.prepart_future is None:
                self.prepart_future = self.thread_pool_executor.submit(
                    ragdoll.pre_partition_graph,
                    dist_info.world_size,
                    self.full_n_nodes,
                    self.full_graph.indptr,
                    self.full_graph.indices,
                    self.mini_graph_info,
                )
            prepart_info = self.prepart_future.result()
            self.prepart_future = None
            end = time.time()
            logger.info("Prepartition done, waiting time: {}s".format(end - start))
            start = time.time()
            sg_n, sg_xadj, sg_adjncy = ragdoll.load_prepart_info(prepart_info)
            end = time.time()
            logger.info("Load prepartition done, time: {}s".format(end - start))
        else:
            # sg_n, sg_xadj, sg_adjncy = ragdoll.partition_graph(0, [], [])
            sg_n, sg_xadj, sg_adjncy = ragdoll.load_prepart_info(None)

        sg_e = sg_xadj[sg_n]
        # assert sg_e == len(sg_adjncy)
        # assert sg_n + 1 == len(sg_xadj)
        # for u in sg_adjncy:
        #     assert u >= 0 and u < sg_n
        # for i in range(sg_n):
        #     assert sg_xadj[i + 1] >= sg_xadj[i]

        edge_data = np.ones([sg_e])
        print("Building csr matrix")
        subgraph = scipy.sparse.csr_matrix(
            (edge_data, sg_adjncy, sg_xadj), shape=[sg_n, sg_n]
        )
        print("Build csr matrix done")

        local_n_nodes = ragdoll.get_local_n_nodes()
        n_nodes = sg_n

        g = DGLGraph(subgraph)
        g = g.to("cuda")

        return g, n_nodes, local_n_nodes

    def _schedule_feat_transfer_plan(self, gid2pid, num_local_nodes, gid2lid):
        # participants是整个恢复组的人数
        # warning: 这里的schedule是根据分区决定的，需要发送和接受的时候根据做转换
        old_nparts = self.prev_num_local_nodes.shape[0]
        new_nparts = num_local_nodes.shape[0]

        send_schedule = [[None for _ in range(new_nparts)] for _ in range(old_nparts)]
        scatter_schedule = [
            [None for _ in range(old_nparts)] for _ in range(new_nparts)
        ]
        for i in range(old_nparts):
            all_old_part_gids = torch.nonzero(self.prev_gid2pid == i).view(-1)
            for j in range(new_nparts):
                # 获取老分区中属于新分区的所有节点的全局node id
                transfer_nodes_gid = all_old_part_gids[gid2pid[all_old_part_gids] == j]
                # 获取老分区i中属于新分区j的所有节点在老分区中的lid
                transfer_nodes_old_part_lid = self.prev_gid2lid[transfer_nodes_gid]
                # 获取新分区中的对应节点的lid
                transfer_nodes_new_part_lid = gid2lid[transfer_nodes_gid]
                send_schedule[i][j] = transfer_nodes_old_part_lid
                scatter_schedule[j][i] = transfer_nodes_new_part_lid

        self.prev_gid2pid = gid2pid
        self.prev_num_local_nodes = num_local_nodes
        self.prev_gid2lid = gid2lid
        return send_schedule, scatter_schedule

    def _load_graph_feat_sync(self, dist_info, n_nodes, local_n_nodes, feat_size):
        # ! 这里默认了0号进程始终有对应的数据
        if dist_info.rank == 0:
            print("feature shape is", self.full_features.shape)
            print("labels shape is", self.full_labels.shape)
            print("train mask shape is", self.full_train_mask.shape)

            features = ragdoll.dispatch_float(
                self.full_features, feat_size, n_nodes, no_remote=1
            )[: local_n_nodes * feat_size]
            labels = ragdoll.dispatch_int(self.full_labels, 1, n_nodes, no_remote=1)[
                :local_n_nodes
            ]
            train_mask = ragdoll.dispatch_int(
                self.full_train_mask, 1, n_nodes, no_remote=1
            )[:local_n_nodes]
        else:
            features = ragdoll.dispatch_float(None, feat_size, n_nodes, no_remote=1)[
                : local_n_nodes * feat_size
            ]
            labels = ragdoll.dispatch_int(None, 1, n_nodes, no_remote=1)[:local_n_nodes]
            train_mask = ragdoll.dispatch_int(None, 1, n_nodes, no_remote=1)[
                :local_n_nodes
            ]

        features = np.reshape(features, [-1, feat_size])
        features = pad_to_128(features)
        labels = np.reshape(labels, [-1])
        train_mask = np.reshape(train_mask, [-1])

        print(f"Rank{dist_info.rank} feature shape is", features.shape)
        print(f"Rank{dist_info.rank} labels shape is", labels.shape)
        print(f"Rank{dist_info.rank} train mask shape is", train_mask.shape)

        return features, labels, train_mask

    def handle_start_training(self, op: DaemonOperation):
        # 情况：可能是新任务开始，也可能是需要恢复的任务开始，
        # 首先需要分图，这一步的时候应当只有新的参与已经建立了新的连接
        # 其次建立recovery distinfo
        # 由0进程完成所有传输规划部分
        # 0进程要能获取所有的local_idx_info
        # 0进程完成规划：每个进程从谁那里拿数据，拿多少数据（甚至分阶段）
        # 0进程分发schedule
        # 开始传输
        # 其次需要在新老设备间完成数据传输，并完成ddp部分的恢复
        # 最后销毁recovery distinfo，继续setup dist
        self.start_training = True
        self.running = True
        self.stop_reason = TrainerStopReason.UNKNOWN
        rec_flag, recovery_schedule, dist_info = op.args

        if dist_info.rank == 0:
            assert not rec_flag or recovery_schedule.recovery_dist_info.rank == 0, (
                "Rank 0 must also be rank 0 in recovery setup group"
            )
            assert (
                not rec_flag
                or recovery_schedule.agent_type == DDPRecoveryType.COMMON_AGENTS
            ), "Rank 0 must be a both agent in recovery setup group"
        else:
            assert not rec_flag or recovery_schedule.recovery_dist_info.rank != 0, (
                "Non-rank 0 must not be rank 0 in recovery setup group"
            )

        # 初始化dgcl恢复组
        need_participate_in_dgcl = (
            recovery_schedule.agent_type != DDPRecoveryType.DEPREACATED_AGENTS
            if rec_flag
            else True
        )
        if need_participate_in_dgcl:
            self.setup_dgcl_dist(
                dist_info
            )  # fixme: maybe stuck here when doing scale down/have old agents shutting down since old ones also try to init dgcl dist
            self.dgcl_inited = True
            g, n_nodes, local_n_nodes = self._partation_graph_and_distribute(dist_info)

        # 初始化torch恢复组
        if rec_flag:
            setup_dist_info = recovery_schedule.recovery_dist_info
        else:
            setup_dist_info = copy.deepcopy(dist_info)
            setup_dist_info.master_port += 3

        rank_in_setup = setup_dist_info.rank
        rank_in_training = dist_info.rank

        if self.distributed_store is None:
            # since gnn always use 0 as master, we also use dist_info as dist store source
            self.distributed_store = dist.TCPStore(  # pyright: ignore
                dist_info.master_addr,
                dist_info.master_port,
                is_master=dist_info.rank == 0,
            )
            logger.debug(
                f"[{dist_info.rank}] Dist store initialized, master addr: {dist_info.master_addr}, port: {dist_info.master_port}"
            )
        recover_dist_store = dist.PrefixStore(
            f"setup_{dist_info.recovery_count}_", self.distributed_store
        )
        dist.init_process_group(
            store=recover_dist_store,
            rank=setup_dist_info.rank,
            world_size=setup_dist_info.world_size,
        )
        comm_group = dist.new_group()  # nccl will init at 1st comm op
        logger.debug("Training recovery group setup done")

        # 初始化setup rank map
        if rec_flag:
            setup_rank_map = recovery_schedule.setup_rank_map
            assert setup_rank_map[0]["old"] == 0 and setup_rank_map[0]["new"] == 0 # 
        else:   # 新训练任务创建默认映射
            setup_rank_map = {
                i: {"old": -1, "new": i} for i in range(1, setup_dist_info.world_size)
            }
            setup_rank_map[0] = {"old": 0, "new": 0}    # 奇怪的设定，保证恢复映射时 idx 0 的old和new都是0

        # 助手函数
        def find_setup_rank_by_oldrank(old_rank):
            for setup_rank, rank_info in setup_rank_map.items():
                if rank_info["old"] == old_rank:
                    return setup_rank
            raise Exception(f"Old rank {old_rank} not found in setup rank map")

        def find_setup_rank_by_newrank(new_rank):
            for setup_rank, rank_info in setup_rank_map.items():
                if rank_info["new"] == new_rank:
                    return setup_rank
            raise Exception(f"New rank {new_rank} not found in setup rank map")

        # 恢复图数据-构建schedule并
        if rank_in_training == 0:
            assert need_participate_in_dgcl
            gid2pid, num_local_nodes, gid2lid = ragdoll.graph_detailed_info(
                self.full_n_nodes, dist_info.world_size
            )
            global_send_schedule, global_scatter_schedule = (
                self._schedule_feat_transfer_plan(gid2pid, num_local_nodes, gid2lid)
            )
            self.prev_gid2pid = gid2pid
            self.prev_num_local_nodes = num_local_nodes
            self.prev_gid2lid = gid2lid
            logger.debug("Build schedule complete")

            # 将每个schedule tensor的长度写到dist store里
            length_info = {
                "send_info": [
                    [
                        len(global_send_schedule[i][j])
                        for j in range(len(global_send_schedule[i]))
                    ]
                    for i in range(len(global_send_schedule))
                ],
                "scatter_info": [
                    [
                        len(global_scatter_schedule[i][j])
                        for j in range(len(global_scatter_schedule[i]))
                    ]
                    for i in range(len(global_scatter_schedule))
                ],
            }
            recover_dist_store.set("length_info", json.dumps(length_info))
            print("Rank", rank_in_training, "set length_info", length_info)
            dist.barrier()
        else:
            dist.barrier()
            length_info = json.loads(recover_dist_store.get("length_info"))
        logger.debug("Get schedule metadata complete")

        # 恢复图数据-传输schedule
        local_send_schedule = []
        local_scatter_schedule = []
        if rank_in_training == 0:
            req_list = []
            for i in range(len(global_send_schedule)):
                setup_rank_of_old_part_i = find_setup_rank_by_oldrank(i)
                if setup_rank_of_old_part_i == 0:
                    local_send_schedule = global_send_schedule[i]
                    continue
                for j in range(len(global_send_schedule[i])):
                    logger.debug(
                        f"send {global_send_schedule[i][j].shape} to {setup_rank_of_old_part_i}, tag {1000 + j}"
                    )
                    req = dist.P2POp(
                        dist.isend,
                        global_send_schedule[i][j].cuda(),
                        setup_rank_of_old_part_i,
                        group=comm_group,
                        tag=1000 + j,
                    )
                    req_list.append(req)

            for i in range(len(global_scatter_schedule)):
                setup_rank_of_new_part_i = find_setup_rank_by_newrank(i)
                if setup_rank_of_new_part_i == 0:
                    local_scatter_schedule = global_scatter_schedule[i]
                    continue
                for j in range(len(global_scatter_schedule[i])):
                    logger.debug(
                        f"send {global_scatter_schedule[i][j].shape} to {setup_rank_of_new_part_i}, tag {2000 + j}"
                    )
                    req = dist.P2POp(
                        dist.isend,
                        global_scatter_schedule[i][j].cuda(),
                        setup_rank_of_new_part_i,
                        group=comm_group,
                        tag=2000 + j,
                    )
                    req_list.append(req)
            if len(req_list) != 0:
                reqs = dist.batch_isend_irecv(req_list)
                for req in reqs:
                    req.wait()
        else:
            req_list = []
            old_rank_of_me = setup_rank_map[rank_in_setup]["old"]
            new_rank_of_me = setup_rank_map[rank_in_setup]["new"]
            logger.debug(
                f"setup_rank {rank_in_setup}, old_rank_of_me {old_rank_of_me}, new_rank_of_me {new_rank_of_me}"
            )
            receive_from = find_setup_rank_by_oldrank(0)
            if old_rank_of_me != -1:
                send_len_list = length_info["send_info"][old_rank_of_me]
                for i in range(len(send_len_list)):
                    send_idx_i = torch.empty(
                        [send_len_list[i]], dtype=torch.int32
                    ).cuda()
                    local_send_schedule.append(send_idx_i)
                for i in range(len(send_len_list)):
                    logger.debug(f"receive from {receive_from}, tag {1000 + i}")
                    req = dist.P2POp(
                        dist.irecv,
                        local_send_schedule[i],
                        receive_from,
                        group=comm_group,
                        tag=1000 + i,
                    )
                    req_list.append(req)
            if new_rank_of_me != -1:
                scatter_len_list = length_info["scatter_info"][new_rank_of_me]
                for i in range(len(scatter_len_list)):
                    scatter_idx_i = torch.empty(
                        [scatter_len_list[i]], dtype=torch.int32
                    ).cuda()
                    local_scatter_schedule.append(scatter_idx_i)
                for i in range(len(scatter_len_list)):
                    logger.debug(
                        f"receive from {receive_from}, tag {2000 + i}, size {scatter_len_list[i]}"
                    )
                    req = dist.P2POp(
                        dist.irecv,
                        local_scatter_schedule[i],
                        receive_from,
                        group=comm_group,
                        tag=2000 + i,
                    )
                    req_list.append(req)
            if len(req_list) != 0:
                reqs = dist.batch_isend_irecv(req_list)
                for req in reqs:
                    req.wait()
        dist.barrier()
        logger.debug("Distribute schedule complete")

        req_list = []
        self_to_self_data = []
        if len(local_send_schedule) is not None:
            for i in range(len(local_send_schedule)):
                target_new_setup_rank = find_setup_rank_by_newrank(i)
                if target_new_setup_rank == setup_dist_info.rank:
                    self_to_self_data.append(
                        self.local_features[local_send_schedule[i]].cuda()
                    )
                    self_to_self_data.append(
                        self.local_labels[local_send_schedule[i]].cuda()
                    )
                    self_to_self_data.append(
                        self.local_train_mask[local_send_schedule[i]].cuda()
                    )
                    continue
                logger.debug(
                    f"{setup_dist_info.rank} send {self.local_features[local_send_schedule[i]].shape} to {i} {target_new_setup_rank}, tag {3000 + i}"
                )
                send_data = self.local_features[local_send_schedule[i]].cuda()
                req = dist.P2POp(
                    dist.isend,
                    send_data,
                    target_new_setup_rank,
                    group=comm_group,
                    tag=3000,
                )
                req_list.append(req)
                send_data = self.local_labels[local_send_schedule[i]].cuda()
                req = dist.P2POp(
                    dist.isend,
                    send_data,
                    target_new_setup_rank,
                    group=comm_group,
                    tag=3100,
                )
                req_list.append(req)
                send_data = self.local_train_mask[local_send_schedule[i]].cuda()
                req = dist.P2POp(
                    dist.isend,
                    send_data,
                    target_new_setup_rank,
                    group=comm_group,
                    tag=3200,
                )
                req_list.append(req)

        recv_features_list = []
        recv_labels_list = []
        recv_train_mask_list = []
        if len(local_scatter_schedule) is not None:
            for i in range(len(local_scatter_schedule)):
                src_new_setup_rank = find_setup_rank_by_oldrank(i)
                if src_new_setup_rank == setup_dist_info.rank:
                    recv_features_list.append(self_to_self_data[0])
                    recv_labels_list.append(self_to_self_data[1])
                    recv_train_mask_list.append(self_to_self_data[2])
                    continue
                recv_data = torch.empty(
                    [
                        len(local_scatter_schedule[i]),
                        self.cfg["dataloader"]["feat_size"],
                    ],
                    device="cuda",
                )
                logger.debug(
                    f"{setup_dist_info.rank} receive from {i} {src_new_setup_rank}, tag {3000}, size {len(local_scatter_schedule[i])}"
                )
                req = dist.P2POp(
                    dist.irecv,
                    recv_data,
                    src_new_setup_rank,
                    group=comm_group,
                    tag=3000,
                )
                req_list.append(req)
                recv_features_list.append(recv_data)
                recv_data = torch.empty(
                    [
                        len(local_scatter_schedule[i]),
                        self.cfg["dataloader"]["label_size"],
                    ],
                    dtype=torch.int32,
                    device="cuda",
                )
                req = dist.P2POp(
                    dist.irecv,
                    recv_data,
                    src_new_setup_rank,
                    group=comm_group,
                    tag=3100,
                )
                req_list.append(req)
                recv_labels_list.append(recv_data)
                recv_data = torch.empty(
                    [len(local_scatter_schedule[i])], dtype=torch.int32, device="cuda"
                )
                req = dist.P2POp(
                    dist.irecv,
                    recv_data,
                    src_new_setup_rank,
                    group=comm_group,
                    tag=3200,
                )
                req_list.append(req)
                recv_train_mask_list.append(recv_data)

        if len(req_list) != 0:
            reqs = dist.batch_isend_irecv(req_list)
            for req in reqs:
                req.wait()

        dist.barrier()

        if len(local_scatter_schedule) != 0:
            self.local_features = torch.empty(
                [local_n_nodes, self.cfg["dataloader"]["feat_size"]], device="cuda"
            )
            self.local_labels = torch.empty(
                [local_n_nodes, self.cfg["dataloader"]["label_size"]],
                dtype=torch.int32,
                device="cuda",
            )
            self.local_train_mask = torch.empty(
                [local_n_nodes], dtype=torch.int32, device="cuda"
            )
            for i in range(len(local_scatter_schedule)):
                self.local_features[local_scatter_schedule[i]] = recv_features_list[i]
                self.local_labels[local_scatter_schedule[i]] = recv_labels_list[i]
                self.local_train_mask[local_scatter_schedule[i]] = recv_train_mask_list[
                    i
                ]

        fixed_features = pad_to_128_torch(self.local_features)
        fixed_labels = self.local_labels.clone().to(torch.long)
        fixed_train_mask = self.local_train_mask.clone()

        del self_to_self_data
        del recv_features_list
        del recv_labels_list
        del recv_train_mask_list

        logger.debug("Distribute graph data complete")

        # todo: check if this is good
        if need_participate_in_dgcl:
            # ref_features, ref_labels, ref_train_mask = self._load_graph_feat_sync(
            #     dist_info, n_nodes, local_n_nodes, self.cfg["dataloader"]["feat_size"]
            # )
            # assert torch.equal(fixed_features, torch.from_numpy(ref_features).cuda())
            # assert torch.equal(fixed_labels, torch.from_numpy(ref_labels).cuda())
            # assert torch.equal(
            #     fixed_train_mask, torch.from_numpy(ref_train_mask).cuda()
            # )
            # logger.debug("Verify graph data success")

            # 创建模型
            model_info = {
                "features": fixed_features,
                "labels": fixed_labels,
                "train_mask": fixed_train_mask,
                "n_classes": self.cfg["dataloader"]["n_classes"],
                "n_nodes": n_nodes,
                "local_n_nodes": local_n_nodes,
                "graph": g,
            }
            self.model.update_network(g, n_nodes, local_n_nodes)
        else:
            # 对于那些该退出的进程，我们仍然需要他们参与后面的活动，但是不需要他们参与到模型的训练中
            # check: 这里可能会出现问题，不过出现问题不太可能（，因为理论上进到这里的进程马上就会退出了
            model_info = {
                "features": fixed_features,
                "labels": fixed_labels,
                "train_mask": fixed_train_mask,
                "n_classes": self.cfg["dataloader"]["n_classes"],
                "n_nodes": 0,
                "local_n_nodes": 0,
                "graph": None,
            }

        # setup basic training resources
        self.criterion = self.task_config.get_criterion(extra_data=model_info)

        # setup training cfg
        # self.epoch = 0
        # self.step = 0
        # self.target_epochs = self.cfg["target_epochs"]

        # setup dataloader
        self.setup_dataloader(self.task_config, dist_info, extra_info=model_info)

        # recover training
        if rec_flag:
            self.recover_training(recovery_schedule, recover_dist_store, comm_group)

        # 销毁数据恢复阶段用到的comm group
        dist.barrier()
        dist.destroy_process_group()
        del recover_dist_store
        logger.debug("Shutting down recovery comm group")

        # setup dist
        if self.running:
            self.setup_dist(dist_info)

        self.tx.put(
            TrainerOperation(
                TrainerOperation.OpCode.TrainingUnpaused,
                (),
            )
        )

    def setup_elastic_data(self, master_device_id=0):
        for idx, param in enumerate(self.model.parameters()):
            assert isinstance(param, torch.Tensor)
            name = f"param_{idx}"
            data = param
            distrib_scheme = ElasDistribScheme(None, 1, {0: [master_device_id]}) # FIX: 使用 Master Device ID 标识
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
                        distrib_scheme = ElasDistribScheme(None, 1, {0: [master_device_id]}) # FIX: 使用 Master Device ID 标识
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

    def handle_pre_scale(self, op: DaemonOperation):
        assert op.op_code == DaemonOperation.OpCode.PreScaleTask
        prescale_info = op.args[0]
        self.prepart_future = self.thread_pool_executor.submit(
            ragdoll.pre_partition_graph,
            prescale_info.n_agents,
            self.full_n_nodes,
            self.full_graph.indptr,
            self.full_graph.indices,
            self.mini_graph_info,
        )

    def handle_query_pre_scale_status(self, op: DaemonOperation):
        status = "not started"
        if self.prepart_future is not None:
            if self.prepart_future.done():
                status = "finished"
            else:
                status = "in_process"

        self.tx.put(
            TrainerOperation(
                TrainerOperation.OpCode.UpdatePreScaleStatus,
                (status,),
            )
        )
