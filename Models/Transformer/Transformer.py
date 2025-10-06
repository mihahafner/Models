# Transformer.py
# Minimal encoder–decoder Transformer + forecaster wrapper (PyTorch)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utilities
# ---------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to embeddings."""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)        # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1), :]


def make_causal_mask(T: int, device=None) -> torch.Tensor:
    """Lower-triangular mask for autoregressive self-attention."""
    m = torch.tril(torch.ones(T, T, device=device))
    # shape (1, 1, T, T) to broadcast across (B, H, T, T)
    return m.unsqueeze(0).unsqueeze(0)


# ---------------------------
# Core attention
# ---------------------------

class Attention(nn.Module): #Layer 1
    """Multi-head scaled dot-product attention with optional mask.
                    +-------------+
        Query  ---> |             |
        Key    ---> |  Attention  | ---> Context Vector
        Value  ---> |             |
                    +-------------+  
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, H, T, Dh)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, Dh) -> (B, T, D)
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(self, q, k, v, mask: torch.Tensor | None = None):
        # q/k/v: (B, T, D)
        Q = self._split_heads(self.q_proj(q))
        K = self._split_heads(self.k_proj(k))
        V = self._split_heads(self.v_proj(v))

        # (B, H, Tq, Tk)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            # mask expected broadcastable to (B, H, Tq, Tk); 0 = block, 1 = keep
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V                                # (B, H, Tq, Dh)
        out = self._combine_heads(out)                # (B, Tq, D)
        return self.o_proj(out)


# ---------------------------
# Transformer blocks
# ---------------------------

class EncoderBlock(nn.Module): #Layer 2
    """
    Each encoder block:
        Has one self-attention layer.
        Then a feed-forward network (FFN).
        Uses residual connections + layer normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = Attention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None):
        x = self.norm1(x + self.drop(self.attn(x, x, x, mask=src_mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderBlock(nn.Module): #Layer 3
    """
    Each decoder block:
        First self-attends to its own output (causal mask → no “peeking” ahead).
        Then attends to the encoder output.
        Followed by a feed-forward network.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = Attention(d_model, n_heads)
        self.cross_attn = Attention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
    ):
        # autoregressive self-attention
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, mask=tgt_mask)))
        # cross-attention over encoder memory
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, mask=mem_mask)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


# ---------------------------
# Stacked Transformer
# ---------------------------

class MiniTransformer(nn.Module): #Layer 4
    """A compact encoder-decoder Transformer stack (no embeddings inside).
        This class glues the encoder and decoder stacks together:
        +-------------------------------+
        |         MiniTransformer       |
        |-------------------------------|
        |  [Encoders]   [Decoders]      |
        |                               |
        |  src -> Encoder -> memory ->  |
        |  tgt -> Decoder(memory) -> y  |
        +-------------------------------+  
    """
    def __init__(self, d_model=64, n_heads=4, d_ff=256, n_enc=2, n_dec=2, dropout=0.1):
        super().__init__()
        self.enc_layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_enc)]
        )
        self.dec_layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_dec)]
        )

    def forward(
        self,
        src: torch.Tensor,           # (B, Ts, D)
        tgt: torch.Tensor,           # (B, Tt, D)
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Encoder
        for layer in self.enc_layers:
            src = layer(src, src_mask=src_mask)
        memory = src

        # Decoder
        x = tgt
        for layer in self.dec_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, mem_mask=mem_mask)
        return x


# ---------------------------
# Wrapper for time-series / generic seq2seq
# ---------------------------

class Seq2SeqForecaster(nn.Module): #Layer 5: Seq2SeqForecaster (Wrapper)
    """
    Wraps MiniTransformer with input/output projections and positional encodings.

    in_dim:  number of input features per time step (source)
    out_dim: number of output features per time step (target)

    Responsibilities:
        Converts raw input/output features into the model’s internal dimension (d_model).
        Adds positional encodings (so the model knows “where” each timestep is).
        Uses the core MiniTransformer to predict sequences.
        Converts model outputs back to numeric form.
    """
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_enc: int = 2,
        n_dec: int = 2,
        dropout: float = 0.1,
        max_len: int = 10000,
    ):
        super().__init__()
        self.src_proj = nn.Linear(in_dim, d_model)
        self.tgt_proj = nn.Linear(out_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.core = MiniTransformer(d_model, n_heads, d_ff, n_enc, n_dec, dropout)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(
        self,
        src: torch.Tensor,    # (B, Ts, in_dim)
        tgt_in: torch.Tensor, # (B, Tt, out_dim) teacher-forced inputs
        use_causal_mask: bool = True,
    ) -> torch.Tensor:
        device = src.device
        src = self.pos_enc(self.src_proj(src))
        tgt = self.pos_enc(self.tgt_proj(tgt_in))

        src_mask = None
        tgt_mask = make_causal_mask(tgt.size(1), device=device) if use_causal_mask else None
        mem_mask = None  # usually None; can be used to restrict cross-attention

        y = self.core(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, mem_mask=mem_mask)
        return self.out_proj(y)

    @torch.no_grad()
    def autoregressive(self, src: torch.Tensor, first_token: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Roll out predictions step-by-step (greedy).
        src: (B, Ts, in_dim)
        first_token: (B, 1, out_dim) initial decoder input (e.g., last known value)
        steps: number of future steps to predict

        Given history + first token:
            Loop for N steps:
                1. Predict next token
                2. Append it to decoder input
                3. Repeat
        """
        self.eval()
        device = src.device
        src_proj = self.pos_enc(self.src_proj(src))
        cur = self.pos_enc(self.tgt_proj(first_token))
        outs = []

        for _ in range(steps):
            tgt_mask = make_causal_mask(cur.size(1), device=device)
            y = self.core(src_proj, cur, tgt_mask=tgt_mask)  # (B, T, D)
            step = self.out_proj(y[:, -1:, :])               # (B, 1, out_dim)
            outs.append(step)
            cur = self.pos_enc(torch.cat([cur, self.tgt_proj(step)], dim=1))

        return torch.cat(outs, dim=1)  # (B, steps, out_dim)


# ---------------------------
# Quick self-test
# ---------------------------

if __name__ == "__main__":
    B, Ts, Tt, in_dim, out_dim = 2, 10, 6, 3, 1
    src = torch.randn(B, Ts, in_dim)
    tgt_in = torch.randn(B, Tt, out_dim)

    model = Seq2SeqForecaster(in_dim=in_dim, out_dim=out_dim, d_model=64, n_heads=4, d_ff=128)
    y = model(src, tgt_in)                 # teacher-forced forward
    print("Forward:", y.shape)             # (B, Tt, out_dim)

    first = tgt_in[:, :1, :]
    y_future = model.autoregressive(src, first, steps=5)
    print("Autoregressive:", y_future.shape)  # (B, 5, out_dim)
