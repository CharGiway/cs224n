"""
Inference for the vanilla Transformer — English → German translation.

Usage:
    python inference.py --checkpoint checkpoints/best.pt --input "hello world"
    python inference.py --checkpoint checkpoints/best.pt --input_file sentences.txt --output_file translations.txt
    python inference.py --checkpoint checkpoints/best.pt --interactive
"""

import argparse
import os
import sys
import torch

from model import Transformer, SinusoidalPositionalEncoding
from train import tokenize_multi30k


def load_model(checkpoint_path, device):
    """Load model and vocabularies from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    src_vocab = ckpt['src_vocab']
    tgt_vocab = ckpt['tgt_vocab']
    a = ckpt['args']

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=a['d_model'],
        n_layers=a['n_layers'],
        n_heads=a['n_heads'],
        d_ff=a['d_ff'],
        dropout=a['dropout'],
        max_len=128,
        pad_idx=0,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    id2tgt = {v: k for k, v in tgt_vocab.items()}
    return model, src_vocab, tgt_vocab, id2tgt


def greedy_decode(model, src_ids, tgt_vocab, device, max_len=128):
    """Greedy decoding for a single source sequence."""
    bos = tgt_vocab['<bos>']
    eos = tgt_vocab['<eos>']

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    enc_out = model.encode(src)
    src_mask = model.make_src_mask(src, model.pad_idx)

    ys = torch.full((1, 1), bos, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model.decode(ys, enc_out, src_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == eos:
            break
    return ys[0]


def beam_search_decode(model, src_ids, tgt_vocab, device,
                       beam_size: int = 5, max_len: int = 128, alpha: float = 0.6):
    """Beam search decoding."""
    bos = tgt_vocab['<bos>']
    eos = tgt_vocab['<eos>']

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    enc_out = model.encode(src)
    src_mask = model.make_src_mask(src, model.pad_idx)

    # Each beam: (sequence, log_prob, finished)
    beams = [(torch.tensor([bos], device=device), 0.0, False)]

    for _ in range(max_len):
        new_beams = []
        for seq, score, done in beams:
            if done:
                new_beams.append((seq, score, done))
                continue

            inp = seq.unsqueeze(0)
            logits = model.decode(inp, enc_out, src_mask)
            next_logits = logits[0, -1, :]          # (vocab_size,)
            log_probs = torch.log_softmax(next_logits, dim=-1)

            topk_scores, topk_ids = torch.topk(log_probs, beam_size)
            for s, tid in zip(topk_scores, topk_ids):
                new_seq = torch.cat([seq, tid.unsqueeze(0)])
                new_score = score + s.item()
                finished = (tid.item() == eos)
                new_beams.append((new_seq, new_score, finished))

        # Prune: keep top beam_size
        new_beams.sort(key=lambda x: -x[1] / (len(x[0]) ** alpha))
        beams = new_beams[:beam_size]

        if all(d for _, _, d in beams):
            break

    best_seq, best_score, _ = beams[0]
    return best_seq


def translate(model, src_vocab, tgt_vocab, id2tgt, sentence: str,
              device, beam_size: int = 1, max_len: int = 128) -> str:
    """Translate a single English sentence to German."""
    tokens = tokenize_multi30k(sentence)
    src_ids = [src_vocab.get(w, 1) for w in tokens]  # 1 = <unk>

    if beam_size == 1:
        result_ids = greedy_decode(model, src_ids, tgt_vocab, device, max_len)
    else:
        result_ids = beam_search_decode(model, src_ids, tgt_vocab, device,
                                        beam_size=beam_size, max_len=max_len)

    result = [id2tgt.get(i.item(), '<unk>')
              for i in result_ids
              if i.item() not in (tgt_vocab['<bos>'], tgt_vocab['<eos>'], tgt_vocab['<pad>'])]
    return ' '.join(result)


def main():
    parser = argparse.ArgumentParser(description='Vanilla Transformer inference')
    parser.add_argument('--checkpoint', default='./checkpoints/best.pt',
                        help='Path to trained checkpoint')
    parser.add_argument('--input', type=str, default=None,
                        help='Single sentence to translate')
    parser.add_argument('--input_file', type=str, default=None,
                        help='File with one sentence per line')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file for translations (default: stdout)')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive translation mode')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='Beam size (1 = greedy)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    model, src_vocab, tgt_vocab, id2tgt = load_model(args.checkpoint, device)
    print(f"  Source vocab: {len(src_vocab)}   Target vocab: {len(tgt_vocab)}")
    print(f"  Beam size: {args.beam_size}")
    print()

    def _translate(text):
        return translate(model, src_vocab, tgt_vocab, id2tgt, text, device,
                         beam_size=args.beam_size)

    # ── Single input ─────────────────────────────────────────────────────
    if args.input:
        result = _translate(args.input)
        print(f"  EN: {args.input}")
        print(f"  DE: {result}")

    # ── File input ───────────────────────────────────────────────────────
    elif args.input_file:
        with open(args.input_file, encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]

        translations = [_translate(l) for l in lines]

        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for t in translations:
                    f.write(t + '\n')
            print(f"Wrote {len(translations)} translations to {args.output_file}")
        else:
            for src, tgt in zip(lines, translations):
                print(f"  EN: {src}")
                print(f"  DE: {tgt}")
                print()

    # ── Interactive ──────────────────────────────────────────────────────
    elif args.interactive:
        print("Enter English sentences to translate (Ctrl+C to exit):\n")
        try:
            while True:
                text = input("EN> ").strip()
                if not text:
                    continue
                result = _translate(text)
                print(f"DE> {result}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")

    # ── Default: demo ────────────────────────────────────────────────────
    else:
        demos = [
            "a man is riding a bicycle",
            "two children are playing in the park",
            "the cat sat on the mat",
        ]
        for d in demos:
            result = _translate(d)
            print(f"  EN: {d}")
            print(f"  DE: {result}")
            print()


if __name__ == '__main__':
    main()
