# -------------------------------------------------
# CLASSIFICATION DATASET + TRAINING
# -------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

from utils.config import DEVICE
from utils.logging_utils import logger


class ClusterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.texts)