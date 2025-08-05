from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from ragdoll.torch.ginconv import GINConv

from geth.base.task_config import TaskConfig
from geth.trainer.common import TrainerTypes


class GIN(nn.Module):
    def __init__(
        self,
        g,
        n_nodes,
        local_n_nodes,
        no_remote,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
    ):
        super(GIN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.bnlayers = nn.ModuleList()
        # input layer
        self.layers.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(in_feats, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_hidden),
                ),
                "sum",
                n_nodes,
                local_n_nodes,
                init_eps=0,
                learn_eps=False,
            )
        )
        self.bnlayers.append(nn.BatchNorm1d(n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(n_hidden, n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden, n_hidden),
                    ),
                    "sum",
                    n_nodes,
                    local_n_nodes,
                    init_eps=0,
                    learn_eps=False,
                )
            )
            self.bnlayers.append(nn.BatchNorm1d(n_hidden))
        # output layer
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def update_network(self, g, n_nodes, local_n_nodes):
        self.g = g
        for i in range(len(self.layers)):
            self.layers[i]._n_nodes = n_nodes
            self.layers[i]._local_n_nodes = local_n_nodes

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(self.g, h))
            h = self.bnlayers[i](h)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return F.log_softmax(h, dim=-1)


class GINTaskConfig(TaskConfig):
    def __init__(self):
        self.n_hidden = 256
        self.n_layers = 4
        self.dropout = 0.5
        self.comm_net = False

        # Loss and optimizer
        self.criterion = F.binary_cross_entropy_with_logits
        self.learning_rate = 1e-2
        self.weight_decay = 5e-4
        self.cfg = {
            "target_epochs": 500,
            "dataloader": {
                "dataset": "pubmed",
                "n_classes": 3,
                "feat_size": 500,
                "label_size": 1,
            },
            "comm_pattern": "dgcl",
        }

        self.model = GIN(
            None,
            -1,
            -1,
            True,
            (self.cfg["dataloader"]["feat_size"] + 128 - 1) // 128 * 128,
            self.n_hidden,
            self.cfg["dataloader"]["n_classes"],
            self.n_layers,
        )
        self.model.cuda()

    def get_model(self, extra_data=None) -> torch.nn.Module:
        return self.model

    def get_dataset(self, extra_data=None) -> torch.utils.data.Dataset:
        # a dummy dataset
        return torch.utils.data.TensorDataset(torch.tensor([0]))

    def get_criterion(self, extra_data=None) -> Callable:
        # train_mask = torch.BoolTensor(extra_data["train_mask"]).cuda()
        train_mask = extra_data["train_mask"].bool()

        def get_loss(output, labels):
            return self.criterion(output[train_mask], labels[train_mask].float())

        return get_loss

    def get_optimizer(self, extra_data=None) -> torch.optim.Optimizer:
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        optimizer.zero_grad()
        return optimizer

    def get_cfg(self) -> dict:
        return self.cfg

    def get_trainer_type(self) -> TrainerTypes:
        return TrainerTypes.DDP_GNN_TRAINER


task_config = GINTaskConfig()
