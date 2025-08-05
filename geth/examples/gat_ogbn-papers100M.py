from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from ragdoll.torch.gatconv import GATConv

from geth.base.task_config import TaskConfig
from geth.trainer.common import TrainerTypes


class GAT(nn.Module):
    def __init__(
        self,
        g,
        n_nodes,
        local_n_nodes,
        no_remote,
        in_feats,
        n_hidden,
        num_heads,
        n_classes,
        n_layers,
        activation,
        dropout,
    ):
        super(GAT, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GATConv(
                in_feats,
                n_hidden,
                num_heads,
                n_nodes,
                local_n_nodes,
                apply_gather=True,
                no_remote=True,
                activation=activation,
            )
        )
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GATConv(
                    n_hidden,
                    n_hidden,
                    num_heads,
                    n_nodes,
                    local_n_nodes,
                    apply_gather=True,
                    no_remote=True,
                    activation=activation,
                )
            )
        # output layer
        self.layers.append(
            GATConv(
                n_hidden,
                n_classes,
                num_heads,
                n_nodes,
                local_n_nodes,
                apply_gather=True,
                no_remote=True,
                activation=activation,
            )
        )
        self.dropout = nn.Dropout(p=dropout)

    def update_network(self, g, n_nodes, local_n_nodes):
        self.g = g
        for i in range(len(self.layers)):
            self.layers[i]._n_nodes = n_nodes
            self.layers[i]._local_n_nodes = local_n_nodes

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = h.flatten(1)
                h = self.dropout(h)
            h = layer(self.g, h)
            h = h.mean(1)
        return h


class GATTaskConfig(TaskConfig):
    def __init__(self):
        self.n_hidden = 32
        self.n_layers = 2
        self.dropout = 0.5
        self.num_heads = 4

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 5e-3
        self.weight_decay = 2.5e-4
        self.cfg = {
            "target_epochs": 500,
            "dataloader": {
                "dataset": "ogbn-papers100M",
                "n_classes": 172,
                "feat_size": 128,
            },
            "comm_pattern": "dgcl",
        }

        self.model = GAT(
            None,
            -1,
            -1,
            True,
            (self.cfg["dataloader"]["feat_size"] + 128 - 1) // 128 * 128,
            self.n_hidden,
            self.num_heads,
            self.cfg["dataloader"]["n_classes"],
            self.n_layers,
            F.elu,
            self.dropout,
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
            return self.criterion(output[train_mask], labels[train_mask])

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


task_config = GATTaskConfig()
