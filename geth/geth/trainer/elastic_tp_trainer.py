import json
from typing import List

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed import ReduceOp

from geth.base.operation import DaemonOperation
from geth.data.comm_manager import CommManager, TopoManager
from geth.data.dataloader import ElasticDataLoader
from geth.data.elas_data import (
    ElasDataManager,
    ElasDataType,
    ElasDistribScheme,
    ElasSlicedTensorData,
)
from geth.data.sampler import ElasticSequentialSampler
from geth.trainer.base_trainer import GethBaseTrainer
from geth.trainer.elastic_ddp_trainer import DDPRecoveryType
from geth.trainer.training_status import TrainerStopReason


class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, world_size):
        ctx.world_size = world_size
        ctx.rank = dist.get_rank()

        # 进行all_gather操作
        gathered_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_list, tensor)
        gathered = torch.cat(gathered_list, dim=-1)

        # 保存输入用于反向传播
        ctx.save_for_backward(tensor)
        return gathered

    @staticmethod
    def backward(ctx, grad_output):
        (tensor,) = ctx.saved_tensors
        world_size = ctx.world_size
        rank = ctx.rank

        # 计算梯度
        grad_input = grad_output.chunk(world_size, dim=-1)[rank]

        # 确保形状匹配
        assert grad_input.shape == tensor.shape, (
            f"grad_input.shape != tensor.shape: {grad_input.shape} != {tensor.shape}"
        )

        return grad_input, None


class TPLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device = "cuda",
        tp_size: int = 1,
        have_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.weight = torch.nn.Parameter(
            torch.randn(out_features // tp_size, in_features, device=device)
        )
        if have_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_features // tp_size, device=device)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor):
        print(f"TPLinear forward: {x.shape} {self.weight.shape}")
        print(
            f"TPLinear forward: {self.in_features}, {self.out_features} {self.tp_size}"
        )
        local_result = x @ self.weight.t()
        if self.bias is not None:
            local_result += self.bias

        # start_comm_time = time.time()
        if self.tp_size > 1:
            result = AllGatherFunction.apply(local_result, self.tp_size)
        else:
            result = local_result
        # end_comm_time = time.time()

        return result

    def update_tp_size(self, tp_size: int):
        self.tp_size = tp_size

    def __repr__(self):
        return f"TPLinear(in_features={self.in_features}, out_features={self.out_features}, tp_size={self.tp_size})"


# 初始化时候修改linear层，后续update tpsize不修改参数，在elasdata里面修改
def init_model_tp_size(
    model: torch.nn.Module,
    tp_size: int,
    rank: int,
    tp_modules: List[str],
    name_prefix: str = "",
):
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Linear):
            # 获取原始线性层的参数
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None

            new_layer = TPLinear(
                in_features,
                out_features,
                device=child.weight.device,
                tp_size=tp_size,
                have_bias=bias,
            )
            new_layer.weight.data.copy_(
                child.weight.data[
                    (out_features // tp_size * rank) : (
                        out_features // tp_size * (rank + 1)
                    ),
                    :,
                ]
            )
            if bias:
                new_layer.bias.data.copy_(
                    child.bias.data[
                        (out_features // tp_size * rank) : (
                            out_features // tp_size * (rank + 1)
                        )
                    ]
                )
            setattr(model, name, new_layer)
            tp_modules[name_prefix + name] = new_layer
        else:
            init_model_tp_size(
                child, tp_size, rank, tp_modules, name_prefix + name + "."
            )
    return model


def update_model_tp_size(model: torch.nn.Module, world_size: int):
    for name, child in list(model.named_children()):
        if isinstance(child, TPLinear):
            child.update_tp_size(world_size)
        else:
            update_model_tp_size(child, world_size)


class GethElasticTPTrainer(GethBaseTrainer):
    def init_worker(self, rank=-1, world_size=1):
        # setup basic training resources
        self.model = self.task_config.get_model()
        self.criterion = self.task_config.get_criterion()
        self.cfg = self.task_config.get_cfg()
        # setup training cfg
        self.epoch = 0
        self.step = 0
        self.target_epochs = self.cfg["target_epochs"]
        tp_modules = {}
        # 这里时因为对于伸缩中新启动的进程worldsize为0，所以需要特殊处理
        # 后面会重新设置，现在就不管了
        tp_rank = rank
        tp_size = world_size
        if tp_size == 0:
            tp_size = 1
            tp_rank = 0
        init_model_tp_size(self.model, tp_size, tp_rank, tp_modules)
        self.setup_elastic_data(tp_modules)

    def setup_dist(self, dist_info):
        dist.init_process_group(
            init_method=dist_info.get_init_url(),
            rank=dist_info.rank,
            world_size=dist_info.world_size,
        )
        dist.barrier()
        update_model_tp_size(self.model, dist_info.world_size)
        del self.optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        logger.debug(f"Rank {dist_info.rank} setup dist done")

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
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def setup_dataloader(self, task_config, dist_info, extra_info=None):
        dataset = task_config.get_dataset()
        cfg = task_config.get_cfg()["dataloader"]
        sampler = ElasticSequentialSampler(
            dataset,  # type: ignore[arg-type]
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
        else:
            self.apply_elastic_data()

    def setup_elastic_data(self, tp_modules: List[str]):
        for idx, (param_name, param) in enumerate(self.model.named_parameters()):
            assert isinstance(param, torch.Tensor)
            name = f"param_{param_name}"
            data = param

            module_name = ".".join(param_name.split(".")[:-1])
            param_suffix_name = param_name.split(".")[-1]
            # print(tp_modules)
            if module_name in tp_modules:
                tp_size = tp_modules[module_name].tp_size
                distrib_scheme = ElasDistribScheme(
                    [0], tp_size, {i: [i] for i in range(tp_size)}
                )
                if param_suffix_name == "weight":
                    full_shape = (
                        tp_modules[module_name].out_features,
                        tp_modules[module_name].in_features,
                    )
                elif param_suffix_name == "bias":
                    full_shape = (tp_modules[module_name].out_features,)
                else:
                    raise ValueError(f"Unknown param suffix name: {param_suffix_name}")

                def update_module_hook(
                    in_data,
                    name=name,
                    module_name=module_name,
                    param_suffix_name=param_suffix_name,
                ):
                    # print(f"update_module_hook: {name} {in_data.shape} {param_suffix_name}", flush=True)
                    if param_suffix_name == "weight":
                        tp_modules[module_name].weight = torch.nn.Parameter(in_data)
                    elif param_suffix_name == "bias":
                        tp_modules[module_name].bias = torch.nn.Parameter(in_data)

                self.elas_data_manager.register_tp_tensor_data(
                    distrib_scheme,
                    ElasDataType.MODEL_PARAM,
                    data,
                    full_shape,
                    name,
                    update_module_hook,
                )
            else:
                distrib_scheme = ElasDistribScheme(None, 1, {0: [0]})
                self.elas_data_manager.register_tensor_data(
                    distrib_scheme, ElasDataType.MODEL_PARAM, data, name
                )
        assert self.optimizer is None or isinstance(self.optimizer, torch.optim.SGD)
        # todo: support more
        # for gid, group in enumerate(self.optimizer.param_groups):
        #     for pid, p in enumerate(group["params"]):
        #         if isinstance(p, torch.Tensor):
        #             state = self.optimizer.state[p]
        #             keys_sorted = sorted(state.keys())
        #             for k in keys_sorted:
        #                 name = f"optim_{gid}_{pid}_{k}"
        #                 data = state[k]
        #                 distrib_scheme = ElasDistribScheme(None, 1, {0: [0]})
        #                 self.elas_data_manager.register_tensor_data(
        #                     distrib_scheme, ElasDataType.OPTIMIZER_STATE, data, name
        #                 )
        #         else:
        #             logger.error("Unknown param:", p)

        # self.elas_data_manager.summary()

    def apply_elastic_data(self):
        for name, data in self.elas_data_manager.data_dict.items():
            if isinstance(data, ElasSlicedTensorData):
                data.post_transfer_hook()

    def apply_new_elastic_schedule(
        self,
        elas_data_manager: ElasDataManager,
        recovery_schedule,
        comm_manager: CommManager,
    ):
        for name, data in elas_data_manager.data_dict.items():
            if isinstance(data, ElasSlicedTensorData):
                old_scheme = recovery_schedule.get_old_distib_scheme_tp(data)
                data.distrib_scheme = old_scheme
                new_scheme = recovery_schedule.get_new_distib_scheme_tp(data)
                # print(f"apply_new_elastic_schedule: {name} {old_scheme} {new_scheme}")
            else:
                new_scheme = recovery_schedule.get_new_distib_scheme(data)
            assert new_scheme is not None, f"Elastic data {name} has no new scheme!"
            data.change_distib_scheme(new_scheme, comm_manager)
