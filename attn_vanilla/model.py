"""
Vanilla Transformer — "Attention Is All You Need" (Vaswani et al., 2017)

Features:
  - Sinusoidal positional encoding (fixed, not learned)
  - Multi-head scaled dot-product attention
  - Encoder-decoder architecture
  - Label smoothing loss

Reference: https://arxiv.org/abs/1706.03762
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sinusoidal Positional Encoding ──────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()       # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                                # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                             # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ── Scaled Dot-Product Attention ────────────────────────────────────────────

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v: (batch, n_heads, seq_len, d_k)
    mask   : (batch, 1, seq_len, seq_len)  or broadcastable
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn


# ── Multi-Head Attention ────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections + split heads
        q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        out, _ = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.W_o(self.dropout(out))


# ── Position-wise Feed-Forward Network ──────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ── Encoder Layer ───────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # Self-attention sub-layer (pre-norm variant for stability)
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        # FFN sub-layer
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.pos_encoding(self.embedding(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ── Decoder Layer ───────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), enc_out, enc_out, src_mask))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.pos_encoding(self.embedding(x))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


# ── Full Transformer ────────────────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1, max_len: int = 5000,
                 pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying: share the decoder embedding and output projection weights
        self.decoder.embedding.weight = self.output_proj.weight

        # Xavier / Glorot init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_src_mask(src, pad_idx):
        # (batch, 1, 1, src_len) — True where not padding
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt, pad_idx):
        batch_size, tgt_len = tgt.size()
        # Padding mask
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
        # Look-ahead mask (causal)
        tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)
        return tgt_pad_mask & tgt_causal_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src, self.pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, self.pad_idx)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.output_proj(dec_out)

    def encode(self, src):
        src_mask = self.make_src_mask(src, self.pad_idx)
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_out, src_mask):
        tgt_mask = self.make_tgt_mask(tgt, self.pad_idx)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.output_proj(dec_out)


# ── Label Smoothing Cross Entropy ───────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        """
        logits: (batch * seq_len, vocab_size)
        target: (batch * seq_len,)
        """
        vocab_size = logits.size(-1)
        true_dist = logits.clone().detach()
        true_dist.fill_(self.smoothing / (vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0  # no smoothing on pad
        mask = (target == self.pad_idx)
        true_dist[mask, self.pad_idx] = 1  # pad targets are 100% pad

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(true_dist * log_probs).sum(dim=-1)
        loss = loss[mask == 0]  # ignore pad positions
        return loss.mean()


# ── Simple tokenizer wrapper (word-level BPE stubs) ─────────────────────────

def build_vocab_from_sentences(sentences, min_freq=2):
    """Build a simple word-level vocabulary from a list of tokenized sentences."""
    word_freq = {}
    for sent in sentences:
        for word in sent:
            word_freq[word] = word_freq.get(word, 0) + 1

    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    idx = 4
    for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1
    return vocab
