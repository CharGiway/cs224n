"""
Train the vanilla Transformer on Multi30k English→German translation.

Usage:
    python train.py                          # default settings
    python train.py --epochs 20 --batch_size 64
    python train.py --d_model 256 --n_layers 4 --n_heads 8

Outputs:
    checkpoints/best.pt    — best validation BLEU model
    checkpoints/last.pt    — last epoch model
    checkpoints/log.txt    — training log
"""

import argparse
import math
import os
import sys
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import (
    Transformer,
    LabelSmoothingLoss,
    build_vocab_from_sentences,
)

# ── Config defaults (paper: base model) ─────────────────────────────────────

CONFIG = dict(
    d_model=256,        # paper uses 512; 256 is faster for Multi30k
    n_layers=4,         # paper uses 6
    n_heads=8,
    d_ff=512,           # paper uses 2048
    dropout=0.1,
    label_smoothing=0.1,
    max_len=128,
    lr=1e-4,
    betas=(0.9, 0.98),
    eps=1e-9,
    warmup_steps=4000,
    epochs=20,
    batch_size=64,
    grad_accum_steps=1,
    clip_grad=1.0,
    seed=42,
)

# ── Tokenization ────────────────────────────────────────────────────────────

def tokenize_multi30k(line: str, lang: str = 'en'):
    """Tokenize a line into words (space-separated + punctuation split)."""
    return line.strip().lower().split()


# ── Dataset ─────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_vocab, tgt_vocab, max_len=128):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.pairs = []

        with open(src_path, encoding='utf-8') as f_src, \
             open(tgt_path, encoding='utf-8') as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                src_tokens = tokenize_multi30k(src_line)
                tgt_tokens = tokenize_multi30k(tgt_line)
                if len(src_tokens) <= max_len and len(tgt_tokens) <= max_len:
                    self.pairs.append((src_tokens, tgt_tokens))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.pairs[idx]
        src_ids = [self.src_vocab.get(w, 1) for w in src_tokens]
        # Add <bos> and <eos>
        tgt_ids = [self.tgt_vocab['<bos>']] + [self.tgt_vocab.get(w, 1) for w in tgt_tokens] + [self.tgt_vocab['<eos>']]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_idx=0):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


# ── Training utilities ──────────────────────────────────────────────────────

def get_noam_scheduler(optimizer, d_model, warmup_steps):
    """Learning rate schedule from paper: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))"""
    d_model = float(d_model)
    warmup_steps = float(warmup_steps)

    def lr_lambda(step):
        step = max(1.0, float(step))
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, criterion, optimizer, device, grad_accum=1):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start = time.time()

    for i, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        # tgt_input = tgt[:, :-1], tgt_output = tgt[:, 1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)          # (batch, tgt_len-1, vocab)
        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(logits, tgt_output)
        loss = loss / grad_accum
        loss.backward()

        total_loss += loss.item() * grad_accum * tgt_output.size(0)
        total_tokens += tgt_output.size(0)

        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_grad'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    elapsed = time.time() - start
    return total_loss / max(total_tokens, 1), elapsed


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(logits, tgt_output)
        total_loss += loss.item() * tgt_output.size(0)
        total_tokens += tgt_output.size(0)

    return total_loss / max(total_tokens, 1)


# ── BLEU (simplified for validation monitoring) ─────────────────────────────

def compute_corpus_bleu(model, dataloader, tgt_vocab, device, max_len=128):
    """Greedy decode + corpus-level BLEU with smoothing."""
    model.eval()
    id2word = {v: k for k, v in tgt_vocab.items()}
    bos = tgt_vocab['<bos>']
    eos = tgt_vocab['<eos>']

    references = []
    hypotheses = []

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Reference (excluding <bos> and <eos>)
        for t in tgt:
            ref = [id2word.get(i.item(), '<unk>') for i in t if i.item() not in (0, bos, eos)]
            references.append([ref])

        # Greedy decode
        enc_out = model.encode(src)
        src_mask = model.make_src_mask(src, model.pad_idx)
        ys = torch.full((src.size(0), 1), bos, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = model.decode(ys, enc_out, src_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos).all():
                break

        for y in ys:
            hyp = [id2word.get(i.item(), '<unk>') for i in y if i.item() not in (0, bos, eos)]
            hypotheses.append(hyp)

    # Simplified BLEU (no sentence-level brevity penalty; corpus-level)
    bleu = corpus_bleu(references, hypotheses)
    return bleu


def ngram_counts(tokens, n):
    c = Counter()
    for i in range(len(tokens) - n + 1):
        c[tuple(tokens[i:i + n])] += 1
    return c


def corpus_bleu(references, hypotheses, max_n=4):
    """Compute corpus-level BLEU score (with smoothing)."""
    precisions = []
    for n in range(1, max_n + 1):
        ref_counts = Counter()
        hyp_counts = Counter()
        for refs, hyp in zip(references, hypotheses):
            hyp_counts += ngram_counts(hyp, n)
            # Closest reference length
            best_ref = min(refs, key=lambda r: abs(len(r) - len(hyp)))
            ref_counts += ngram_counts(best_ref, n)

        total = sum(hyp_counts.values())
        match = sum((ref_counts & hyp_counts).values())
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(match / (total + 1e-10))

    # Brevity penalty
    hyp_len = sum(len(h) for h in hypotheses)
    ref_len = sum(min(len(r) for r in refs) for refs in references)
    bp = math.exp(1 - ref_len / hyp_len) if hyp_len < ref_len and hyp_len > 0 else 1.0

    geo_mean = math.exp(sum(math.log(p + 1e-10) for p in precisions) / max_n)
    return bp * geo_mean


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train vanilla Transformer')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--d_model', type=int, default=CONFIG['d_model'])
    parser.add_argument('--n_layers', type=int, default=CONFIG['n_layers'])
    parser.add_argument('--n_heads', type=int, default=CONFIG['n_heads'])
    parser.add_argument('--d_ff', type=int, default=CONFIG['d_ff'])
    parser.add_argument('--dropout', type=float, default=CONFIG['dropout'])
    parser.add_argument('--lr', type=float, default=CONFIG['lr'])
    parser.add_argument('--warmup', type=int, default=CONFIG['warmup_steps'])
    parser.add_argument('--grad_accum', type=int, default=CONFIG['grad_accum_steps'])
    parser.add_argument('--max_len', type=int, default=CONFIG['max_len'])
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("Vanilla Transformer — Training")
    print(f"  Device: {device}")
    print(f"  d_model={args.d_model}  layers={args.n_layers}  heads={args.n_heads}  d_ff={args.d_ff}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  warmup={args.warmup}")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    train_src_path = os.path.join(args.data_dir, 'train.en')
    train_tgt_path = os.path.join(args.data_dir, 'train.de')
    val_src_path   = os.path.join(args.data_dir, 'val.en')
    val_tgt_path   = os.path.join(args.data_dir, 'val.de')

    for p in [train_src_path, train_tgt_path, val_src_path, val_tgt_path]:
        if not os.path.exists(p):
            sys.exit(f"Data file not found: {p}\nRun: bash download_data.sh")

    print("\nBuilding vocabularies...")
    train_src_tokens = [tokenize_multi30k(l) for l in open(train_src_path, encoding='utf-8')]
    train_tgt_tokens = [tokenize_multi30k(l) for l in open(train_tgt_path, encoding='utf-8')]

    src_vocab = build_vocab_from_sentences(train_src_tokens, min_freq=2)
    tgt_vocab = build_vocab_from_sentences(train_tgt_tokens, min_freq=2)
    print(f"  Source vocab: {len(src_vocab)}   Target vocab: {len(tgt_vocab)}")

    # ── Datasets & DataLoaders ───────────────────────────────────────────
    train_dataset = TranslationDataset(train_src_path, train_tgt_path,
                                       src_vocab, tgt_vocab, max_len=args.max_len)
    val_dataset   = TranslationDataset(val_src_path, val_tgt_path,
                                       src_vocab, tgt_vocab, max_len=args.max_len)
    print(f"  Train pairs: {len(train_dataset)}   Val pairs: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4)

    # ── Model, loss, optimizer, scheduler ────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=CONFIG['max_len'],
        pad_idx=0,
    ).to(device)

    criterion = LabelSmoothingLoss(smoothing=CONFIG['label_smoothing'], pad_idx=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=CONFIG['betas'], eps=CONFIG['eps'])
    scheduler = get_noam_scheduler(optimizer, args.d_model, args.warmup)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ── Training loop ────────────────────────────────────────────────────
    log_file = os.path.join(args.save_dir, 'log.txt')
    best_bleu = 0.0
    patience = 20

    with open(log_file, 'w') as log:
        log.write("epoch\ttrain_loss\tval_loss\tval_bleu\ttime\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, elapsed = train_epoch(model, train_loader, criterion,
                                          optimizer, device, args.grad_accum)
        val_loss = validate(model, val_loader, criterion, device)
        val_bleu = compute_corpus_bleu(model, val_loader, tgt_vocab, device,
                                       max_len=CONFIG['max_len'])

        print(f"Epoch {epoch:2d} | train loss: {train_loss:.4f} | "
              f"val loss: {val_loss:.4f} | BLEU: {val_bleu:.4f} | {elapsed:.1f}s")

        with open(log_file, 'a') as log:
            log.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\t{val_bleu:.6f}\t{elapsed:.1f}\n")

        # Save best & last
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'args': vars(args),
            'bleu': val_bleu,
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pt'))

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))
            print(f"  → new best BLEU: {best_bleu:.4f}")

    print(f"\nTraining complete. Best BLEU: {best_bleu:.4f}")


if __name__ == '__main__':
    main()
