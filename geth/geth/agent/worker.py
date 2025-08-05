import importlib.util
import os
import sys
import types

from loguru import logger
from torch.multiprocessing import Process, SimpleQueue

from geth.base.task_config import TaskConfig
from geth.trainer.elastic_ddp_trainer import GethElasticDDPTrainer


def module_from_file(module_name, file_path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"load client module {module_name} from file failed!"
    assert spec.loader is not None, (
        f"load client module {module_name} from file failed!"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    logger.debug(f"load client module {module} from file success!")
    return module


def setup_trainer_proc(
    worker_id,
    device_id,
    task_name: str,
    config_file: str,
    trainer_rx: SimpleQueue,
    trainer_tx: SimpleQueue,
):
    os.environ["GETH_DEVICE"] = str(device_id)
    import torch

    torch.cuda.set_device(device_id)

    config_module = module_from_file(task_name, config_file)
    task_config = config_module.task_config
    assert isinstance(task_config, TaskConfig)
    trainer_type = task_config.get_trainer_type()
    trainer = None
    match trainer_type:
        case trainer_type.DDP_TRAINER:
            trainer = GethElasticDDPTrainer(
                worker_id, trainer_rx, trainer_tx, task_config
            )
        case trainer_type.DDP_GNN_TRAINER:
            from geth.trainer.elastic_ddp_gnn_trainer import GethElasticDDPGNNTrainer

            trainer = GethElasticDDPGNNTrainer(
                worker_id, trainer_rx, trainer_tx, task_config
            )
        case trainer_type.TP_TRAINER:
            from geth.trainer.elastic_tp_trainer import GethElasticTPTrainer

            trainer = GethElasticTPTrainer(
                worker_id, trainer_rx, trainer_tx, task_config
            )

    trainer.train_loop()


class GethWorker:
    def __init__(self, worker_id, device_id, task_name: str, config_file: str):
        self.rx = SimpleQueue()
        self.tx = SimpleQueue()
        self.worker_id = worker_id
        self.device_id = device_id
        self.task_name = task_name
        self.config_file = config_file
        self.trainer_proc = Process(
            target=setup_trainer_proc,
            args=(
                self.worker_id,
                self.device_id,
                self.task_name,
                self.config_file,
                self.tx,
                self.rx,
            ),
        )
        self.in_training_proc = False
        self.in_pre_scale_proc = False
        self.trainer_proc.start()

    def stop(self, timeout=5):
        self.trainer_proc.join(timeout)
        ret_code = self.trainer_proc.exitcode
        if ret_code is None:
            logger.warning(
                f"Forcefully quit trainer proc due to unable to join after timeout of {timeout}s"
            )
            self.trainer_proc.kill()
            self.trainer_proc.join()
            ret_code = self.trainer_proc.exitcode

        logger.debug(f"Worker finished with ret_code = {ret_code}")
