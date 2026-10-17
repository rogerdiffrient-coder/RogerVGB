"""Roger Spark: prompt-conditioned decoder-only Transformer."""
import torch
from torch import nn

# 123.51M parameters with the browser-compatible 192-token context.
TARGET_PARAMETERS = 123_509_504
DEFAULT_WIDTH = 640
DEFAULT_LAYERS = 28
DEFAULT_HEADS = 10
DEFAULT_MAX_SEQ = 128
DEFAULT_VOCAB_SIZE = 6144
DEFAULT_FFN_DIMS = [2048] * DEFAULT_LAYERS

class CausalBlock(nn.Module):
    def __init__(self, width, heads, ff_dim, max_seq, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, width), nn.Dropout(dropout),
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq, max_seq, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        length = x.shape[1]
        y = self.norm1(x)
        # need_weights=False lets PyTorch use its optimized scaled-dot-product attention path.
        x = x + self.attn(y, y, y, attn_mask=self.causal_mask[:length, :length], need_weights=False)[0]
        return x + self.ff(self.norm2(x))

class RogerSpark(nn.Module):
    def __init__(self, vocab_size=DEFAULT_VOCAB_SIZE, width=DEFAULT_WIDTH,
                 layers=DEFAULT_LAYERS, heads=DEFAULT_HEADS, max_seq=DEFAULT_MAX_SEQ,
                 ffn_dims=None, dropout=0.0):
        super().__init__()
        ffn_dims = list(ffn_dims or [2048] * layers)
        if len(ffn_dims) != layers:
            raise ValueError(f"Expected {layers} FFN sizes, got {len(ffn_dims)}")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.config = dict(
            vocab_size=vocab_size, width=width, layers=layers, heads=heads,
            max_seq=max_seq, ffn_dims=ffn_dims, dropout=dropout, pad_id=0, eos_id=1,
            parameter_target=TARGET_PARAMETERS,
        )
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(max_seq, width)
        self.blocks = nn.ModuleList([
            CausalBlock(width, heads, ff_dim, max_seq, dropout) for ff_dim in ffn_dims
        ])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight
        self.apply(self._init)
        actual = self.count_parameters()
        if actual != TARGET_PARAMETERS:
            raise ValueError(f"Architecture has {actual:,} parameters; expected {TARGET_PARAMETERS:,}")

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids):
        length = input_ids.shape[1]
        if length > self.config["max_seq"]:
            input_ids = input_ids[:, -self.config["max_seq"]:]
            length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        x = self.token(input_ids) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))
