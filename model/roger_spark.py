"""Roger 0.1 Spark: a decoder-only Transformer implemented from scratch.

No pretrained weights, Ollama, hosted model APIs, or external inference runtime.
The model architecture and weights are RogerVGB's own. Training is a separate step.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SparkConfig:
    vocab_size: int = 24000
    context_length: int = 512
    d_model: int = 1024
    n_layers: int = 18
    n_heads: int = 16
    dropout: float = 0.0
    model_name: str = "Roger 0.1 Spark"

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        # PyTorch's scaled_dot_product_attention uses an optimized kernel when available.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model, bias=False),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class RogerSpark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie input/output embeddings to reduce parameter count and memory.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        batch, length = input_ids.shape
        if length > self.config.context_length:
            raise ValueError(f"Input has {length} tokens; context limit is {self.config.context_length}.")
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            cropped = input_ids[:, -self.config.context_length:]
            logits, _ = self(cropped)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k < logits.size(-1):
                    values, _ = torch.topk(logits, top_k)
                    logits[logits < values[:, [-1]]] = -float("inf")
                probabilities = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids


def tiny_config():
    """Small configuration for testing the training pipeline quickly."""
    return SparkConfig(vocab_size=512, context_length=128, d_model=128, n_layers=4, n_heads=4,
                       dropout=0.0, model_name="Roger 0.1 Spark (tiny test)")
