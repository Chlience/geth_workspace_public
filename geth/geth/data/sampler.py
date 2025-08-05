import math
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Sized

import torch
import torch.utils
import torch.utils.data
from torch.utils.data import Sampler


class ElasticSampler(Sampler, ABC):
    @abstractmethod
    def save_state(self, prefetch_num: int) -> Any:
        pass

    @abstractmethod
    def load_state(self, state: Any):
        pass


class ElasticSequentialSampler(ElasticSampler):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        super().__init__()
        self.data_source = data_source
        self._internal_cnt = 0
        self._target_len = len(self.data_source)

    def save_state(self, prefetch_num: int):
        if self._internal_cnt <= prefetch_num:
            self._internal_cnt = prefetch_num
        return {"internal_cnt": self._internal_cnt - prefetch_num}

    def load_state(self, state):
        self._internal_cnt = state["internal_cnt"]

    def __iter__(self) -> Iterator[int]:
        while self._internal_cnt < self._target_len:
            self._internal_cnt += 1
            yield self._internal_cnt - 1
        self._internal_cnt = 0

    def __len__(self) -> int:
        return self._target_len


class ElasticRandomSampler(ElasticSampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.current_generator = generator
        self._internal_cnt = 0
        self._data_seq = None

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self._data_seq is None:
            self._data_seq = []
            if self.replacement:
                for _ in range(self.num_samples // 32):
                    self._data_seq.extend(
                        torch.randint(
                            high=n, size=(32,), dtype=torch.int64, generator=generator
                        ).tolist()
                    )
                self._data_seq.extend(
                    torch.randint(
                        high=n,
                        size=(self.num_samples % 32,),
                        dtype=torch.int64,
                        generator=generator,
                    ).tolist()
                )
            else:
                for _ in range(self.num_samples // n):
                    self._data_seq.extend(
                        torch.randperm(n, generator=generator).tolist()
                    )
                self._data_seq.extend(
                    torch.randperm(n, generator=generator).tolist()[
                        : self.num_samples % n
                    ]
                )

        while self._internal_cnt < n:
            self._internal_cnt += 1
            yield self._data_seq[self._internal_cnt - 1]

        self._internal_cnt = 0
        self._data_seq = None

    def __len__(self) -> int:
        return self.num_samples

    def save_state(self, prefetch_num: int):
        if self._internal_cnt <= prefetch_num:
            self._internal_cnt = prefetch_num
        generator_state = None
        if self.generator is not None:
            generator_state = self.generator.get_state()
        return {
            "internal_cnt": self._internal_cnt - prefetch_num,
            "data_seq": self._data_seq,
            "generator_state": generator_state,
        }

    def load_state(self, state):
        self._internal_cnt = state["internal_cnt"]
        self._data_seq = state["data_seq"]
        if state["generator_state"] is not None:
            if self.generator is None:
                self.generator = torch.Generator()
            self.generator.set_state(state["generator_state"])


class ElasticDistributedSampler(ElasticSampler):
    def __init__(
        self,
        dataset: Sized,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        assert num_replicas is not None
        assert rank is not None
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self._internal_cnt = self.rank

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        while self._internal_cnt < n:
            self._internal_cnt += self.num_replicas
            yield indices[self._internal_cnt - self.num_replicas]

        self._internal_cnt = self.rank

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def save_state(self, prefetch_num: int):
        if self._internal_cnt <= prefetch_num * self.num_replicas:
            self._internal_cnt = prefetch_num * self.num_replicas
        return {
            "internal_cnt_base": self._internal_cnt
            - prefetch_num * self.num_replicas
            - self.rank,
        }

    def load_state(self, state):
        self._internal_cnt = state["internal_cnt_base"] + self.rank
