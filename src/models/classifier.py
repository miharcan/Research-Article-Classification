# -------------------------------------------------
# CLASSIFICATION DATASET + TRAINING
# -------------------------------------------------
import torch
from torch.utils.data import Dataset


class ClusterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.labels = labels
        # Pre-tokenize once so we don't re-tokenize every sample each epoch.
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        label = self.labels[idx]
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.labels)
