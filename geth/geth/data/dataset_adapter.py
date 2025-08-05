from torch.utils.data import Dataset


class GethDatasetAdapter(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "input": {"args": [self.dataset[idx][0]], "kwargs": {}},
            "target": {"args": [self.dataset[idx][1]], "kwargs": {}},
        }
