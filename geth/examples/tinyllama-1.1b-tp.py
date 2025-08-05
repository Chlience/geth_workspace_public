import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from geth.base.task_config import TaskConfig
from geth.trainer.common import TrainerTypes

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            [self.texts[idx]],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        item = {key: val[0] for key, val in encoding.items()}
        item["labels"] = item["input_ids"].clone()
        return {
            "input": {"args": [], "kwargs": item},
            "target": {"args": [], "kwargs": {}},
        }

    def __len__(self):
        return len(self.texts)


def format_instruction(example):
    """将数据集中的指令、输入和输出格式化为模型可接受的格式"""
    instruction = example["instruction"]
    input_text = example["input"] if example["input"] else ""
    output = example["output"]

    if input_text:
        formatted_text = (
            f"<|user|>\n{instruction}\n\n{input_text}<|assistant|>\n{output}"
        )
    else:
        formatted_text = f"<|user|>\n{instruction}<|assistant|>\n{output}"

    return {"formatted_text": formatted_text}


class TinyLlamaTPTaskConfig(TaskConfig):
    def __init__(self):
        self.batch_size = 20
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Using TinyLlama as a smaller alternative
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir="/workspace/huggingface-cache/tokenizer"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, cache_dir="/workspace/huggingface-cache/model"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        dataset_obj = load_dataset(
            "vicgalle/alpaca-gpt4",
            split="train",
            cache_dir="/workspace/huggingface-cache/dataset",
        )
        formatted_dataset = dataset_obj.map(format_instruction)
        texts = formatted_dataset["formatted_text"]
        self.dataset = TextDataset(texts, self.tokenizer)

        # Loss and optimizer
        self.criterion = lambda x: x.loss
        self.learning_rate = 0.001
        self.cfg = {
            "target_epochs": 500,
            "dataloader": {
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
        raise Exception("Optimizer is not supported in TP trainer")

    def get_cfg(self) -> dict:
        return self.cfg

    def get_trainer_type(self) -> TrainerTypes:
        return TrainerTypes.TP_TRAINER


task_config = TinyLlamaTPTaskConfig()
