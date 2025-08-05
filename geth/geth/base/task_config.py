from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.utils
import torch.utils.data

from geth.trainer.common import TrainerTypes


class TaskConfig(ABC):
    @abstractmethod
    def get_model(self, extra_data: Any = None) -> torch.nn.Module:
        pass

    @abstractmethod
    def get_dataset(self, extra_data: Any = None) -> torch.utils.data.Dataset:
        pass

    @abstractmethod
    def get_criterion(self, extra_data: Any = None) -> Callable:
        pass

    @abstractmethod
    def get_optimizer(self, extra_data: Any = None) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_cfg(self) -> dict:
        pass

    @abstractmethod
    def get_trainer_type(self) -> TrainerTypes:
        pass
