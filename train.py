from model import SimpleGPT2Module, GPT2Config
import pytorch_lightning as L
import torch

from data import SimpleGPT2DataModule

torch.set_float32_matmul_precision("medium")
torch.set_default_dtype(torch.bfloat16)

dm = SimpleGPT2DataModule("./input.txt", batch_size=4, seq_len=16, num_workers=2)
model = SimpleGPT2Module(GPT2Config())

trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=20,
    default_root_dir="./checkpoints",
)
trainer.fit(model, dm)
