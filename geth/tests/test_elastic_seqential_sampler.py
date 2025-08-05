from torch.utils.data import Dataset

from geth.data.sampler import ElasticSequentialSampler


class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TestElasticSequentialSampler:
    def test_init(self):
        # 测试初始化方法
        data_source = [1, 2, 3, 4, 5]
        sampler = ElasticSequentialSampler(data_source)
        assert sampler.data_source == data_source
        assert sampler._internal_cnt == 0
        assert sampler._target_len == len(data_source)

    def test_save_state(self):
        # 测试save_state方法
        data_source = [1, 2, 3, 4, 5]
        sampler = ElasticSequentialSampler(data_source)
        sampler._internal_cnt = 3
        state = sampler.save_state(0)
        assert state == {"internal_cnt": 3}

    def test_save_state_with_prefetch(self):
        data_source = [1, 2, 3, 4, 5]
        sampler = ElasticSequentialSampler(data_source)
        sampler._internal_cnt = 3
        state = sampler.save_state(1)
        assert state == {"internal_cnt": 2}

    def test_load_state(self):
        # 测试load_state方法
        data_source = [1, 2, 3, 4, 5]
        sampler = ElasticSequentialSampler(data_source)
        state = {"internal_cnt": 3}
        sampler.load_state(state)
        assert sampler._internal_cnt == 3

    def test_iter(self):
        # 测试__iter__方法
        data_source = [1, 2, 3, 4, 5]
        sampler = ElasticSequentialSampler(data_source)
        assert list(sampler) == [0, 1, 2, 3, 4]
        assert list(sampler) == [0, 1, 2, 3, 4]

    def test_len(self):
        # 测试__len__方法
        data_source = [1, 2, 3, 4, 5]
        sampler = ElasticSequentialSampler(data_source)
        assert len(sampler) == 5
        sampler._internal_cnt = 3
        assert len(sampler) == 5

    def test_save_and_load(self):
        data_source = [1, 2, 3, 4, 5, 6]
        sampler = ElasticSequentialSampler(data_source)
        truth = list(sampler)

        # first iter some and save
        sampler = ElasticSequentialSampler(data_source)
        iterator = iter(sampler)
        target = []
        for idx in range(2):
            target.append(next(iterator))
        state = sampler.save_state(0)
        # load saved iter state
        sampler = ElasticSequentialSampler(data_source)
        sampler.load_state(state)
        # continue iterating
        target += list(sampler)

        assert truth == target

        new_target = list(sampler)
        assert new_target == truth
