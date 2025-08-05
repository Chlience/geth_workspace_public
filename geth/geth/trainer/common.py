from enum import Enum


class TrainerTypes(Enum):
    DDP_TRAINER = 0
    DDP_GNN_TRAINER = 1
    TP_TRAINER = 2
