from enum import Enum
from typing import Optional

from torch.utils.data import DataLoader

from geth.data.sampler import ElasticSampler


class DataloaderType(Enum):
    STANDALONE_DATALOADER = 0
    STANDALONE_RANDOM_DATALOADER = 1
    DISTRIBUTED_RANDOM_DATALOADER = 2


class ElasticDataLoader(DataLoader):
    def __init__(self, *args, sampler: Optional[ElasticSampler] = None, **kwargs):
        assert isinstance(sampler, ElasticSampler), (
            "ElasticSampler required for ElasticDataloader"
        )

        super().__init__(*args, sampler=sampler, **kwargs)

    def save_dataloader_state(self):
        assert isinstance(self.sampler, ElasticSampler), (
            "ElasticSampler required for ElasticDataloader"
        )
        prefetch_num = 0
        if self.prefetch_factor is not None:
            assert self.batch_size is not None
            # todo: it can be wrong if prefetch reached end of dataset
            prefetch_num = self.batch_size * self.prefetch_factor * self.num_workers
        return self.sampler.save_state(prefetch_num)

    def load_dataloader_state(self, state):
        assert isinstance(self.sampler, ElasticSampler), (
            "ElasticSampler required for ElasticDataloader"
        )
        self.sampler.load_state(state)
