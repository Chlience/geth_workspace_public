from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.utils
from loguru import logger
from torch import optim
from torch.multiprocessing import SimpleQueue
from torch.optim.optimizer import Optimizer

from geth.base.dist_info import DistInfo
from geth.base.operation import DaemonOperation, TrainerOperation
from geth.base.task_config import TaskConfig
from geth.data.dataloader import ElasticDataLoader
from geth.data.elas_data import ElasDataManager
from geth.trainer.training_status import TrainerStopReason, TrainingStatus


class GethRecoveryShedule:
    def __init__(self):
        pass


def transfer_to_device(data: dict, device: torch.device):
    for i in range(len(data["input"]["args"])):
        if isinstance(data["input"]["args"][i], torch.Tensor):
            data["input"]["args"][i] = data["input"]["args"][i].to(device)
    for i in range(len(data["target"]["args"])):
        if isinstance(data["target"]["args"][i], torch.Tensor):
            data["target"]["args"][i] = data["target"]["args"][i].to(device)

    for key, value in data["input"]["kwargs"].items():
        if isinstance(value, torch.Tensor):
            data["input"]["kwargs"][key] = value.to(device)
    for key, value in data["target"]["kwargs"].items():
        if isinstance(value, torch.Tensor):
            data["target"]["kwargs"][key] = value.to(device)

    return data


class GethBaseTrainer(ABC):
    def __init__(
        self,
        worker_id: int,
        rx: SimpleQueue,
        tx: SimpleQueue,
        task_config: TaskConfig,
    ):
        self.worker_id = worker_id
        self.rx = rx
        self.tx = tx
        self.task_config: TaskConfig = task_config
        self.elas_data_manager = ElasDataManager()

        self.model: torch.nn.Module = None  # pyright: ignore
        self.dataloader: ElasticDataLoader = None  # pyright: ignore
        self.criterion: Callable = None  # pyright: ignore
        self.optimizer: Optimizer = None  # pyright: ignore
        self.cfg: dict = {}

        self.epoch = 0
        self.step = 0
        self.target_epochs = 0
        self.running = True
        self.training = True
        self.stop_reason = TrainerStopReason.WAITING
        self.pending_pause = False

        self.recover_data: Any = None
        self.start_training = False

    def init_worker(self, rank=-1, world_size=1, target_epoch=-1):
        # setup basic training resources
        self.model = self.task_config.get_model()
        self.criterion = self.task_config.get_criterion()
        self.optimizer = self.task_config.get_optimizer()
        self.cfg = self.task_config.get_cfg()
        # setup training cfg
        self.epoch = 0
        self.step = 0
        # self.target_epochs = self.cfg["target_epochs"]
        self.target_epochs = target_epoch if target_epoch > 0 else self.cfg["target_epochs"]
        self.init_optimizer_state(self.optimizer)
        self.setup_elastic_data()

    def before_train(self):
        self.handle_msg(
            [
                DaemonOperation.OpCode.StartTraining,
                DaemonOperation.OpCode.StopTraining,
                DaemonOperation.OpCode.FiniWorker,
                DaemonOperation.OpCode.QueryTrainingStatus,
                DaemonOperation.OpCode.PreScaleTask,
                DaemonOperation.OpCode.QueryPreScaleStatus,
            ]
        )

    def after_train(self):
        self.stage_local_training_resource()
        self.teardown_dist()
        logger.info(
            f"Stopped: self.worker_id={self.worker_id}, self.stop_reason={self.stop_reason}"
        )

        if self.pending_pause:
            self.pending_pause = False
            self.tx.put(
                TrainerOperation(
                    TrainerOperation.OpCode.TrainingPaused,
                    (),
                )
            )

    def before_epoch(self):
        self.epoch += 1
        self.step = 0
        # pass

    def after_epoch(self):
        # self.epoch += 1
        # self.step = 0
        logger.debug(f"Epoch: self.epoch = {self.epoch}")
        if self.epoch == self.target_epochs:
            self.running = False
            self.stop_reason = TrainerStopReason.FINISHED

    def before_step(self):
        self.step += 1
        # pass

    def after_step(self):
        # self.step += 1
        logger.debug(
            f"Stat: self.worker_id={self.worker_id}, self.epoch={self.epoch}, self.step={self.step}"
        )
        self.handle_msg(None)
        self.sync_running_status()

    @abstractmethod
    def setup_elastic_data(self):
        pass

    @abstractmethod
    def setup_dist(self, dist_info: DistInfo):
        pass

    @abstractmethod
    def teardown_dist(self):
        pass

    @abstractmethod
    def setup_dataloader(
        self, task_config: TaskConfig, dist_info: DistInfo, extra_info: Any = None
    ):
        pass

    @abstractmethod
    def stage_local_training_resource(self):
        pass

    @abstractmethod
    def recover_training(self, recovery_schedule: GethRecoveryShedule):
        pass

    @abstractmethod
    def sync_running_status(self):
        pass

    def init_optimizer_state(self, optimizer):
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimizer.state[p] = {}  # Initialize an empty state for each parameter

                # For Adam
                if isinstance(optimizer, optim.Adam):
                    optimizer.state[p]["step"] = torch.tensor(0.0, device=p.device)
                    optimizer.state[p]["exp_avg"] = torch.zeros_like(
                        p.data, device=p.device
                    )
                    optimizer.state[p]["exp_avg_sq"] = torch.zeros_like(
                        p.data, device=p.device
                    )

                # For RMSprop
                if isinstance(optimizer, optim.RMSprop):
                    if group["momentum"] > 0:
                        optimizer.state[p]["momentum_buffer"] = torch.zeros_like(
                            p.data, device=p.device
                        )

                # For SGD
                if isinstance(optimizer, optim.SGD):
                    if group["momentum"] > 0:
                        optimizer.state[p]["momentum_buffer"] = torch.zeros_like(
                            p.data, device=p.device
                        )
                    if group["nesterov"]:
                        optimizer.state[p]["velocity"] = torch.zeros_like(
                            p.data, device=p.device
                        )

    def handle_msg(self, wait_for=None):
        if (wait_for is None) and (self.rx.empty()):
            return

        op = None
        while op is None:
            op = self.rx.get()
            assert isinstance(op, DaemonOperation)
            if wait_for is not None and op.op_code not in wait_for:
                logger.error(f"Not expecting such op! {op}")
                op = None

        logger.debug(f"Trainer Got op: {op}")
        match op.op_code:
            case DaemonOperation.OpCode.InitWorker:
                self.handle_init_worker(op)
            case DaemonOperation.OpCode.FiniWorker:
                self.handle_fini_worker(op)
            case DaemonOperation.OpCode.StartTraining:
                self.handle_start_training(op)
            case DaemonOperation.OpCode.StopTraining:
                self.handle_stop_training(op)
            case DaemonOperation.OpCode.QueryTrainingStatus:
                self.handle_query_training_status(op)
            case DaemonOperation.OpCode.PreScaleTask:
                self.handle_pre_scale(op)
            case DaemonOperation.OpCode.QueryPreScaleStatus:
                self.handle_query_pre_scale_status(op)

    def handle_init_worker(self, op: DaemonOperation):
        self.init_worker(rank=op.args[0], world_size=op.args[1], master_device_id=op.args[2], target_epoch=op.args[3])
        self.tx.put(
            TrainerOperation(
                TrainerOperation.OpCode.TrainingInited,
                (),
            )
        )

    def handle_fini_worker(self, op: DaemonOperation):
        self.running = False
        self.training = False
        self.tx.put(
            TrainerOperation(
                TrainerOperation.OpCode.TrainingTerminated,
                (),
            )
        )

    def handle_start_training(self, op: DaemonOperation):
        self.start_training = True
        self.running = True
        self.stop_reason = TrainerStopReason.UNKNOWN

        rec_flag, recovery_schedule, dist_info = op.args
        # setup dataloader
        self.setup_dataloader(self.task_config, dist_info, extra_info=None)

        # recover training
        if rec_flag:
            self.recover_training(recovery_schedule)

        # setup dist
        if self.running:
            self.setup_dist(dist_info)

        self.tx.put(
            TrainerOperation(
                TrainerOperation.OpCode.TrainingUnpaused,
                (),
            )
        )

    def handle_stop_training(self, op: DaemonOperation):
        self.running = False
        self.pending_pause = (
            True  # make sure the proc is really paused (dist torn down) before reply
        )
        self.stop_reason = TrainerStopReason.STOPPED_BY_AGENT

    def handle_query_training_status(self, op: DaemonOperation):
        self.tx.put(
            TrainerOperation(
                TrainerOperation.OpCode.UpdateTrainerStatus,
                (
                    TrainingStatus(
                        self.epoch, self.step, self.running, self.stop_reason
                    ),
                ),
            )
        )

    def handle_pre_scale(self, op: DaemonOperation):
        pass

    def handle_query_pre_scale_status(self, op: DaemonOperation):
        self.tx.put(
            TrainerOperation(
                TrainerOperation.OpCode.UpdatePreScaleStatus,
                ("finished",),
            )
        )

    def train(self):
        self.model.train()
        while self.epoch < self.target_epochs and self.running:
            self.before_epoch()

            for data in self.dataloader:
                if not self.running:
                    break

                self.before_step()

                data = transfer_to_device(
                    data, torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                self.optimizer.zero_grad()
                outputs = self.model(*data["input"]["args"], **data["input"]["kwargs"])
                loss = self.criterion(
                    outputs, *data["target"]["args"], **data["target"]["kwargs"]
                )
                loss.backward()
                self.optimizer.step()

                self.after_step()

            if not self.running:
                break

            self.after_epoch()

    def train_loop(self):
        self.handle_msg([DaemonOperation.OpCode.InitWorker])
        while self.training:
            self.before_train()
            if self.start_training:
                self.train()
                self.after_train()
