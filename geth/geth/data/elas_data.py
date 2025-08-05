from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from loguru import logger

from geth.data.comm_manager import CommManager


class ElasDataType(Enum):
    TRAIN_DATASET = 0
    MODEL_PARAM = 1
    OPTIMIZER_STATE = 2

# 由于 scale 时会创建新的通信组，和以前的 rank 不一定会对应
# 所以需要全局量（worker_id）来标识数据的位置
class ElasDistribScheme:
    def __init__(
        self, axes, nparts, partition_to_worker_map: Dict[int, List[int]]
    ) -> None:
        self.axes = axes
        self.nparts = nparts
        self.partition_to_worker_map = partition_to_worker_map
        self.worker_to_partition_map = {}
        for partition, worker_ids in partition_to_worker_map.items():
            for worker_id in worker_ids:
                self.worker_to_partition_map[worker_id] = partition

    def __repr__(self) -> str:
        return f"ElasDistribScheme(axes={self.axes}, nparts={self.nparts}, partition_to_worker_map={self.partition_to_worker_map}, worker_to_partition_map={self.worker_to_partition_map})"


class ElasData:
    def __init__(
        self,
        data_id: int,
        distrib_scheme: ElasDistribScheme,
        elas_data_type: ElasDataType,
        local_data: Any,
        name: Optional[str] = None,
    ) -> None:
        self.data_id = data_id
        self.name = name
        self.distrib_scheme = distrib_scheme
        self.elas_data_type = elas_data_type
        self.local_data = local_data

    def change_distib_scheme(
        self, new_distrib_scheme: ElasDistribScheme, comm_manager: CommManager
    ) -> None:
        raise Exception("Not implemented!")


class ElasTensorData(ElasData):
    def __init__(
        self,
        data_id: int,
        distrib_scheme: ElasDistribScheme,
        elas_data_type: ElasDataType,
        local_data: torch.Tensor,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(data_id, distrib_scheme, elas_data_type, local_data, name)

    """
    事实上 distrib_scheme 并不会进行修改，永远都留存在原来的 worker 上
    """
    def change_distib_scheme(
        self, new_distrib_scheme: ElasDistribScheme, comm_manager: CommManager
    ):
        self_worker_id = comm_manager.topo_manager.worker_id
        local_data_clone = self.local_data.clone()

        # get whom should this proc send to
        prev_partition = self.distrib_scheme.worker_to_partition_map.get(
            self_worker_id, None
        )
        if prev_partition is not None:
            prev_partition_owners = self.distrib_scheme.partition_to_worker_map[
                prev_partition
            ]
            cur_partition_owners = new_distrib_scheme.partition_to_worker_map.get(
                prev_partition, []
            )
            for cur_owner in cur_partition_owners:
                if cur_owner == self_worker_id:
                    continue
                # ! fixme: this still send for 1st scale down even if no need to send
                best_sender = comm_manager.topo_manager.get_min_cost_peer(
                    cur_owner, prev_partition_owners
                )
                if best_sender == self_worker_id:
                    # logger.debug(f"Worker {self_worker_id} will send ElasTensorData {self.data_id} {self.name} to {cur_owner}")
                    comm_manager.schedule_send(
                        local_data_clone, cur_owner, self.data_id
                    )

        # get whom should this proc recv from
        # ! fixme: for new procs this only recv from proc 0 since when init the default distrib scheme is all data on 0
        new_partition = new_distrib_scheme.worker_to_partition_map.get(
            self_worker_id, None
        )
        if new_partition is not None:
            prev_partition_owners = self.distrib_scheme.partition_to_worker_map[
                new_partition
            ]
            best_sender = comm_manager.topo_manager.get_min_cost_peer(
                self_worker_id, prev_partition_owners
            )
            if best_sender != self_worker_id:
                # logger.debug(f"Worker {self_worker_id} will recv ElasTensorData {self.data_id} {self.name} from {best_sender}")
                comm_manager.schedule_recv(self.local_data, best_sender, self.data_id)


class ElasSlicedTensorData(ElasData):
    def __init__(
        self,
        data_id: int,
        distrib_scheme: ElasDistribScheme,
        elas_data_type: ElasDataType,
        local_data: torch.Tensor,
        full_shape: Tuple[int, ...],
        name: Optional[str] = None,
        after_recomb_hook: Optional[Callable] = None,
    ) -> None:
        super().__init__(data_id, distrib_scheme, elas_data_type, local_data, name)
        self.full_shape = full_shape
        self.after_recomb_hook = after_recomb_hook
        self.recomb_buf = []

    def change_distib_scheme(
        self, new_distrib_scheme: ElasDistribScheme, comm_manager: CommManager
    ):
        assert new_distrib_scheme.axes == self.distrib_scheme.axes, (
            "supports only axis-parallel slicing"
        )
        axis = self.distrib_scheme.axes[0]
        dim_size_to_part = self.full_shape[axis]
        assert dim_size_to_part % self.distrib_scheme.nparts == 0, (
            "old distrib scheme must be a multiple of the full shape"
        )
        assert dim_size_to_part % new_distrib_scheme.nparts == 0, (
            "new distrib scheme must be a multiple of the full shape"
        )
        prev_chunk_size = dim_size_to_part // self.distrib_scheme.nparts
        new_chunk_size = dim_size_to_part // new_distrib_scheme.nparts

        self_worker_id = comm_manager.topo_manager.worker_id
        prev_partition_id = self.distrib_scheme.worker_to_partition_map.get(
            self_worker_id, None
        )
        if prev_partition_id is not None:
            self_prev_start, self_prev_end = (
                prev_chunk_size * prev_partition_id,
                prev_chunk_size * (prev_partition_id + 1),
            )
            for new_part_id in range(new_distrib_scheme.nparts):
                new_start, new_end = (
                    new_chunk_size * new_part_id,
                    new_chunk_size * (new_part_id + 1),
                )
                overlap_start, overlap_end = (
                    max(new_start, self_prev_start),
                    min(new_end, self_prev_end),
                )
                if overlap_end - overlap_start > 0:
                    old_part_owning_workers = (
                        self.distrib_scheme.partition_to_worker_map[prev_partition_id]
                    )
                    new_part_owning_workers = (
                        new_distrib_scheme.partition_to_worker_map[new_part_id]
                    )
                    for cur_owner in new_part_owning_workers:
                        # ! fixme: this still send for 1st scale down even if no need to send
                        best_sender = comm_manager.topo_manager.get_min_cost_peer(
                            cur_owner, old_part_owning_workers
                        )
                        if best_sender == self_worker_id:
                            tensor_slice = self.local_data.narrow(
                                axis,
                                overlap_start - self_prev_start,
                                overlap_end - overlap_start,
                            )
                            comm_manager.schedule_send(
                                tensor_slice, cur_owner, self.data_id
                            )

        # handle recv
        new_partition_id = new_distrib_scheme.worker_to_partition_map.get(
            self_worker_id, None
        )
        if new_partition_id is not None:
            new_start, new_end = (
                new_chunk_size * new_partition_id,
                new_chunk_size * (new_partition_id + 1),
            )
            self_new_start, self_new_end = (
                new_chunk_size * new_partition_id,
                new_chunk_size * (new_partition_id + 1),
            )
            for prev_part_id in range(self.distrib_scheme.nparts):
                prev_start, prev_end = (
                    prev_chunk_size * prev_part_id,
                    prev_chunk_size * (prev_part_id + 1),
                )
                overlap_start, overlap_end = (
                    max(prev_start, self_new_start),
                    min(prev_end, self_new_end),
                )
                if overlap_end - overlap_start > 0:
                    old_part_owning_workers = (
                        self.distrib_scheme.partition_to_worker_map[prev_part_id]
                    )
                    best_sender = comm_manager.topo_manager.get_min_cost_peer(
                        self_worker_id, old_part_owning_workers
                    )
                    tensor_shape = list(self.full_shape)
                    tensor_shape[axis] = overlap_end - overlap_start
                    tensor_slice = torch.zeros(
                        tensor_shape, dtype=self.local_data.dtype, device="cuda"
                    )
                    self.recomb_buf.append(tensor_slice)
                    comm_manager.schedule_recv(tensor_slice, best_sender, self.data_id)
        return

    def post_transfer_hook(self):
        recombed_data = torch.cat(self.recomb_buf, dim=self.distrib_scheme.axes[0])
        self.local_data = recombed_data
        self.recomb_buf = []
        if self.after_recomb_hook is not None:
            self.after_recomb_hook(recombed_data)


class ElasDataManager:
    def __init__(self) -> None:
        self.data_dict = {}
        self.data_dict_by_name = {}
        self.id = 0

    def register_tensor_data(
        self,
        distrib_scheme: ElasDistribScheme,
        elas_data_type: ElasDataType,
        local_data: torch.Tensor,
        name: Optional[str] = None,
    ):
        elas_data = ElasTensorData(
            self.id, distrib_scheme, elas_data_type, local_data, name
        )
        self.data_dict[self.id] = elas_data
        self.data_dict_by_name[name] = elas_data
        self.id += 1

    def register_tp_tensor_data(
        self,
        distrib_scheme: ElasDistribScheme,
        elas_data_type: ElasDataType,
        local_data: torch.Tensor,
        full_shape: Tuple[int, ...],
        name: Optional[str] = None,
        after_recomb_hook: Optional[Callable] = None,
    ):
        elas_data = ElasSlicedTensorData(
            self.id,
            distrib_scheme,
            elas_data_type,
            local_data,
            full_shape,
            name,
            after_recomb_hook,
        )
        self.data_dict[self.id] = elas_data
        self.data_dict_by_name[name] = elas_data
        self.id += 1

    def summary(self):
        logger.debug("ElasDataManager summary:")
        for k, v in self.data_dict.items():
            logger.debug(
                f"data_id: {k}, name: {v.name}, elas_data_type: {v.elas_data_type}, local_data: {v.local_data.dtype} {v.local_data.shape}"
            )
