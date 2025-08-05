from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from ragdoll.torch.graphconv import GraphConv

from geth.base.task_config import TaskConfig
from geth.trainer.common import TrainerTypes


class GCN(nn.Module):
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
        activation,
        dropout,
        comm_net,
    ):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(
                in_feats,
                n_hidden,
                n_nodes,
                local_n_nodes,
                apply_gather=True,
                no_remote=True,
                norm="none",
                activation=activation,
                comm_net=comm_net,
            )
        )
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(
                    n_hidden,
                    n_hidden,
                    n_nodes,
                    local_n_nodes,
                    apply_gather=True,
                    norm="none",
                    activation=activation,
                    comm_net=comm_net,
                )
            )
        # output layer
        self.layers.append(
            GraphConv(
                n_hidden,
                n_classes,
                n_nodes,
                local_n_nodes,
                apply_gather=True,
                no_remote=True,
                norm="none",
                comm_net=comm_net,
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
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


class GCNTaskConfig(TaskConfig):
    def __init__(self):
        self.n_hidden = 512
        self.n_layers = 4
        self.dropout = 0.5
        self.comm_net = False

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-2
        self.weight_decay = 5e-4
        self.cfg = {
            "target_epochs": 500,
            "dataloader": {
                "dataset": "ogbn-arxiv",
                "n_classes": 40,
                "feat_size": 128,
                "label_size": 1,
            },
            "comm_pattern": "dgcl",
        }

        self.model = GCN(
            None,
            -1,
            -1,
            True,
            (self.cfg["dataloader"]["feat_size"] + 128 - 1) // 128 * 128,
            self.n_hidden,
            self.cfg["dataloader"]["n_classes"],
            self.n_layers,
            F.relu,
            self.dropout,
            comm_net=self.comm_net,
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
            return self.criterion(output[train_mask], labels[train_mask].flatten())

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


task_config = GCNTaskConfig()
