import torch
from torch.utils.data import Dataset

from geth.data.sampler import ElasticRandomSampler


class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TestElasticRandomSampler:
    def test_save_and_load(self):
        generator = torch.Generator()
        generator.manual_seed(0)
        data_source = [1, 2, 3, 4, 5, 6]
        sampler = ElasticRandomSampler(data_source, generator=generator)
        truth = list(sampler) + list(sampler)

        # first iter some and save
        generator = torch.Generator()
        generator.manual_seed(0)
        sampler = ElasticRandomSampler(data_source, generator=generator)
        iterator = iter(sampler)
        target = []
        for idx in range(2):
            target.append(next(iterator))
        state = sampler.save_state(0)
        # load saved iter state
        generator = torch.Generator()
        generator.manual_seed(0)
        sampler = ElasticRandomSampler(data_source, generator=generator)
        sampler.load_state(state)
        # continue iterating
        target += list(sampler)
        target += list(sampler)

        assert truth == target
