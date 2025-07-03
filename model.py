from typing import Tuple, Optional
from dataclasses import dataclass
from torch import nn
import math
import torch.nn.functional as F
import pytorch_lightning as L
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.bfloat16)

BATCH_SIZE = 4
NUM_WORKERS = int(os.cpu_count() / 2)
GRAD_ACCUM = 4


@dataclass
class GPT2Config:
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 6
    n_embd: int = 768
    n_head: int = 6


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class GPT2CasualSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.config: GPT2Config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.block_size, config.block_size).view(
                    1, 1, config.block_size, config.block_size
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v

        # Kernelized flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.permute(0, 2, 1, 3).reshape(B, T, C)
        y = self.c_proj(y)
        return y


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GPT2CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SimpleGPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config: GPT2Config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            is_residual = any(
                name in module.__class__.__name__.lower() for name in ["c_proj"]
            )

            std = (2 * self.config.n_layer) ** -0.5 if is_residual else std
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = idx.size()
        loss = None
        assert T <= self.config.block_size, (
            "Cannot forward sequence of length > block_size."
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class SimpleGPT2Module(L.LightningModule):
    def __init__(
        self, config: GPT2Config, lr: float = 3e-4, batch_size: int = BATCH_SIZE
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.model = torch.compile(SimpleGPT2(config))
        self.model = SimpleGPT2(config)
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x, y)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits, loss = self(x, y)
        loss = loss / GRAD_ACCUM
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        self.toggle_optimizer(optimizer)
        # optimizer.zero_grad()
        self.manual_backward(loss)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log("grad_norm", norm, on_step=True, on_epoch=True)
        if (batch_idx + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        self.untoggle_optimizer(optimizer)

        self.log("lr", scheduler.get_last_lr()[0], on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # Final gradient step if any unstepped grads are left
        if any(p.grad is not None for p in self.parameters()):
            self.toggle_optimizer(optimizer)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            self.untoggle_optimizer(optimizer)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        logits, loss = self(x, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        max_lr = 3e-4
        min_lr = max_lr * 0.2
        warmup_steps = 10
        max_steps = 50

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {k: v for k, v in param_dict.items() if v.requires_grad}

        decay_params = [p for p in param_dict.values() if p.ndim >= 2]
        no_decay_params = [p for p in param_dict.values() if p.ndim < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return max_lr * (current_step + 1 / warmup_steps)
            elif current_step > max_steps:
                return min_lr
            else:
                # Cosine decay
                progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return (min_lr / max_lr) + (1 - (min_lr / max_lr)) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def run_inference(self, x: torch.Tensor, max_tokens: int = 128):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            while x.size(1) < max_tokens:
                logits, _ = self.model(x)
                logits = logits[:, -1, :]

                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)

                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)

                x = torch.cat((x, xcol), dim=1)

        return x
