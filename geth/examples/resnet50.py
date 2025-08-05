import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets, models, transforms

from geth.base.task_config import TaskConfig
from geth.data.dataset_adapter import GethDatasetAdapter
from geth.trainer.common import TrainerTypes

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10: 10 classes
num_classes = 10

# Transformations for the input data
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load CIFAR-10 data
train_dataset = datasets.CIFAR10(
    root="./dataset/cifar10", train=True, transform=transform, download=True
)
test_dataset = datasets.CIFAR10(
    root="./dataset/cifar10", train=False, transform=transform, download=True
)


# Define the ResNet50 model
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Replace the last fully connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class Resnet50TaskConfig(TaskConfig):
    def __init__(self):
        self.batch_size = 128
        self.model = ResNet50(num_classes).to(device)

        imagenet_sample_num = 1281167
        self_sample_num = len(train_dataset)
        concat_time = (imagenet_sample_num + self_sample_num - 1) // self_sample_num
        self.dataset = GethDatasetAdapter(
            Subset(
                ConcatDataset([train_dataset] * concat_time), range(imagenet_sample_num)
            )
        )

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.cfg = {
            "target_epochs": 500,
            "dataloader": {
                "random": True,
                "batch_size": self.batch_size,
                "num_workers": 5,
            },
        }

    def get_model(self, extra_data=None) -> torch.nn.Module:
        return self.model

    def get_dataset(self, extra_data=None) -> torch.utils.data.Dataset:
        return self.dataset

    def get_criterion(self, extra_data=None) -> torch.nn.Module:
        return self.criterion

    def get_optimizer(self, extra_data=None) -> torch.optim.Optimizer:
        self.optimizer.zero_grad()
        return self.optimizer

    def get_cfg(self) -> dict:
        return self.cfg

    def get_trainer_type(self) -> TrainerTypes:
        return TrainerTypes.DDP_TRAINER


task_config = Resnet50TaskConfig()
