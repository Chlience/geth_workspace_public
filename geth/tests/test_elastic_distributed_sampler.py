from torch.utils.data import Dataset

from geth.data.sampler import ElasticDistributedSampler


class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TestElasticDistributedSampler:
    def test_basic_iter(self):
        dataset = MockDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        sampler = ElasticDistributedSampler(
            dataset, num_replicas=4, rank=1, shuffle=False
        )
        assert list(sampler) == [1, 5, 9]
        sampler.set_epoch(10)
        assert sampler.epoch == 10
        assert list(sampler) == [1, 5, 9]

    def test_basic_iter_save_and_load(self):
        dataset = MockDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        sampler = ElasticDistributedSampler(
            dataset, num_replicas=3, rank=1, shuffle=False
        )
        truth = list(sampler) + list(sampler)

        iterator = iter(sampler)
        target = []
        for idx in range(3):
            target.append(next(iterator))
        target.pop()
        state = sampler.save_state(prefetch_num=1)
        sampler = ElasticDistributedSampler(
            dataset, num_replicas=3, rank=1, shuffle=False
        )
        sampler.load_state(state)
        target += list(sampler)
        target += list(sampler)

        assert truth == target

    def test_random_iter(self):
        dataset = MockDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        sampler1 = ElasticDistributedSampler(dataset, num_replicas=3, rank=0)
        sampler2 = ElasticDistributedSampler(dataset, num_replicas=3, rank=1)
        sampler3 = ElasticDistributedSampler(dataset, num_replicas=3, rank=2)
        target = []
        target += list(sampler1)
        target += list(sampler2)
        target += list(sampler3)

        assert sorted(target) == list(range(12))

        sampler1.set_epoch(10)
        sampler2.set_epoch(10)
        sampler3.set_epoch(10)

        target = []
        target += list(sampler1)
        target += list(sampler2)
        target += list(sampler3)
        assert sorted(target) == list(range(12))

    def test_random_iter_save_and_load(self):
        dataset = MockDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        sampler = ElasticDistributedSampler(dataset, num_replicas=3, rank=1)
        truth = list(sampler)
        sampler.set_epoch(1)
        truth += list(sampler)

        sampler.set_epoch(0)
        iterator = iter(sampler)
        target = []
        for idx in range(3):
            target.append(next(iterator))
        target.pop()
        state = sampler.save_state(prefetch_num=1)
        sampler = ElasticDistributedSampler(dataset, num_replicas=3, rank=1)
        sampler.load_state(state)
        target += list(sampler)
        sampler.set_epoch(1)
        target += list(sampler)

        assert truth == target

    def test_random_iter_change_replicas(self):
        dataset = MockDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        sampler1 = ElasticDistributedSampler(dataset, num_replicas=3, rank=0)
        sampler2 = ElasticDistributedSampler(dataset, num_replicas=3, rank=1)
        sampler3 = ElasticDistributedSampler(dataset, num_replicas=3, rank=2)

        sampler1.set_epoch(0)
        sampler2.set_epoch(0)
        sampler3.set_epoch(0)
        iterator1 = iter(sampler1)
        iterator2 = iter(sampler2)
        iterator3 = iter(sampler3)
        target = []
        for idx in range(3):
            target.append(next(iterator1))
            target.append(next(iterator2))
            target.append(next(iterator3))
        target.pop()
        target.pop()
        target.pop()

        state = sampler1.save_state(prefetch_num=1)

        sampler1 = ElasticDistributedSampler(dataset, num_replicas=2, rank=0)
        sampler2 = ElasticDistributedSampler(dataset, num_replicas=2, rank=1)
        sampler1.load_state(state)
        sampler2.load_state(state)
        target += list(sampler1)
        target += list(sampler2)

        assert sorted(target) == list(range(len(dataset)))
