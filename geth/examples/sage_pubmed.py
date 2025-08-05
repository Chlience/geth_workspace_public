from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from ragdoll.torch.sageconv import SAGEConv

from geth.base.task_config import TaskConfig
from geth.trainer.common import TrainerTypes


class SAGE(nn.Module):
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
    ):
        super(SAGE, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            SAGEConv(
                in_feats,
                n_hidden,
                "pool",
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
                SAGEConv(
                    n_hidden,
                    n_hidden,
                    "pool",
                    n_nodes,
                    local_n_nodes,
                    apply_gather=True,
                    no_remote=True,
                    activation=activation,
                )
            )
        # output layer
        self.layers.append(
            SAGEConv(
                n_hidden,
                n_classes,
                "pool",
                n_nodes,
                local_n_nodes,
                apply_gather=True,
                no_remote=True,
                activation=activation,
            )
        )

    def update_network(self, g, n_nodes, local_n_nodes):
        self.g = g
        for i in range(len(self.layers)):
            self.layers[i]._n_nodes = n_nodes
            self.layers[i]._local_n_nodes = local_n_nodes

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(self.g, h))
        return F.log_softmax(h, dim=-1)


class SAGETaskConfig(TaskConfig):
    def __init__(self):
        self.n_hidden = 256
        self.n_layers = 4
        self.dropout = 0.5

        # Loss and optimizer
        self.criterion = F.binary_cross_entropy_with_logits
        self.learning_rate = 5e-3
        self.weight_decay = 2.5e-4
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

        self.model = SAGE(
            None,
            -1,
            -1,
            True,
            (self.cfg["dataloader"]["feat_size"] + 128 - 1) // 128 * 128,
            self.n_hidden,
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


task_config = SAGETaskConfig()
