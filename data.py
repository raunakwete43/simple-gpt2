import pytorch_lightning as L
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import os
import tiktoken

BATCH_SIZE = 4
NUM_WORKERS = int(os.cpu_count() / 2)


class GPTTextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size
        self.num_samples = len(tokens) - block_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError("Index out of bounds for dataset")
        if idx < 0:
            idx += self.num_samples
        x = torch.tensor(self.tokens[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(
            self.tokens[idx + 1 : idx + 1 + self.block_size], dtype=torch.long
        )
        return x, y


class SimpleGPT2DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = BATCH_SIZE,
        seq_len: int = 128,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding("gpt2")

    def prepare_data(self):
        with open(self.data_path, "r") as f:
            data = f.read()
        self.tokens = self.enc.encode(data)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            dataset_full = GPTTextDataset(self.tokens[:128], self.seq_len)
            self.train_dataset, self.val_dataset = random_split(
                dataset_full, [0.7, 0.3]
            )

        if stage == "test" or stage is None:
            self.test_dataset = GPTTextDataset(self.tokens[:128], self.seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
