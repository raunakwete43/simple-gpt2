from model import SimpleGPT2Module, GPT2Config
import pytorch_lightning as L
import torch

from data import SimpleGPT2DataModule

torch.set_float32_matmul_precision("medium")

dm = SimpleGPT2DataModule("./input.txt", batch_size=4, seq_len=16, num_workers=2)
model = SimpleGPT2Module(GPT2Config())

trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=5,
    default_root_dir="./checkpoints",
    log_every_n_steps=4,
)
trainer.fit(model, dm)
