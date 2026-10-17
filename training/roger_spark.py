"""Roger Spark: a tiny decoder-only Transformer trained from random initialization.
This is intentionally a small educational model, not a pretrained model.
"""
import torch
from torch import nn
from torch.nn import functional as F

class ByteTokenizer:
    """Byte-level tokenizer: 0 is padding, UTF-8 bytes are shifted by 1, 257 is EOS."""
    pad_id = 0
    eos_id = 257
    vocab_size = 258
    def encode(self, text):
        return [b + 1 for b in text.encode("utf-8")] + [self.eos_id]
    def decode(self, ids):
        data = bytes(max(0, min(255, int(i) - 1)) for i in ids if 1 <= int(i) <= 256)
        return data.decode("utf-8", errors="ignore")

class CausalBlock(nn.Module):
    def __init__(self, width, heads, dropout, max_seq):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout))
        # Cache the causal mask instead of rebuilding it on every training step.
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq, max_seq, dtype=torch.bool), diagonal=1), persistent=False)
    def forward(self, x):
        length = x.shape[1]
        mask = self.causal_mask[:length, :length]
        y = self.norm1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.norm2(x))

class RogerSpark(nn.Module):
    def __init__(self, vocab_size=258, width=128, layers=4, heads=4, max_seq=512, dropout=0.05):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, width=width, layers=layers, heads=heads, max_seq=max_seq, dropout=dropout)
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(max_seq, width)
        self.blocks = nn.ModuleList([CausalBlock(width, heads, dropout, max_seq) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight
        self.apply(self._init)
    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)
    def forward(self, ids):
        _, length = ids.shape
        if length > self.config["max_seq"]: ids = ids[:, -self.config["max_seq"]:]; length = ids.shape[1]
        pos = torch.arange(length, device=ids.device)
        x = self.token(ids) + self.position(pos)[None, :, :]
        for block in self.blocks: x = block(x)
        return self.lm_head(self.norm(x))

def sample_next(logits, temperature=0.8, top_p=0.9):
    temperature = max(0.1, min(1.5, float(temperature)))
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative - sorted_probs >= max(0.1, min(1.0, float(top_p)))
    sorted_probs[remove] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    chosen = torch.multinomial(sorted_probs, 1)
    return sorted_ids.gather(-1, chosen)
