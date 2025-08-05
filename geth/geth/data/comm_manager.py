# type: ignore[attr-defined] # torch.distributed

from typing import Dict, List

import torch
import torch.distributed as dist


class TopoManager:
    def __init__(self, worker_id: int, rank_map: Dict[int, int]) -> None:
        self.worker_id = worker_id
        self.rank_map = rank_map

    def get_rank(self, worker_id: int):
        return self.rank_map[worker_id]

    def get_min_cost_peer(self, worker_id, peers: List[int]):
        if worker_id in peers:
            return worker_id
        return sorted(peers)[0]


class CommManager:
    def __init__(self, topo_manager: TopoManager, comm_group):
        self.topo_manager = topo_manager
        self.comm_group = comm_group
        self.req_list = []

    def _gen_send_recv_tag(self, sender: int, reciever: int, extra_tag: int):
        tag = (sender << 24) | (reciever << 16) | extra_tag
        return tag

    def schedule_send(self, src_data: torch.Tensor, dst_worker: int, extra_tag: int):
        tag = self._gen_send_recv_tag(
            self.topo_manager.worker_id, dst_worker, extra_tag
        )
        dst_rank = self.topo_manager.get_rank(dst_worker)
        # logger.info(f"Send {src_data.shape} from {self.topo_manager.worker_id} to {dst_worker} with tag {tag}")
        req = dist.P2POp(dist.isend, src_data, dst_rank, group=self.comm_group, tag=tag)
        self.req_list.append(req)

    def schedule_recv(self, dst_data: torch.Tensor, src_worker: int, extra_tag: int):
        tag = self._gen_send_recv_tag(
            src_worker, self.topo_manager.worker_id, extra_tag
        )
        src_rank = self.topo_manager.get_rank(src_worker)
        # logger.info(f"Recv {dst_data.shape} from {src_worker} to {self.topo_manager.worker_id} with tag {tag}")
        req = dist.P2POp(dist.irecv, dst_data, src_rank, group=self.comm_group, tag=tag)
        self.req_list.append(req)

    def commit(self):
        p2p_ops = self.req_list
        if len(p2p_ops) != 0:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()
        self.req_list = []
