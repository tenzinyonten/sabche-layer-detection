#!/usr/bin/env python3
"""
eval_particle_rule.py - forbid tsawa labels on quote particles at decode time.

Usage
-----
    python eval_particle_rule.py --model Yontenn/mmbert-tsawa-bio-inv-v2 \
        --dataset ds_v2 --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer

NEG = -1.0e9
O, B, I = 0, 1, 2

# longest first, so ཞེས་པ་སྟེ is matched before ཞེས་
CLOSERS = sorted(["ཞེས་པ་སྟེ", "ཞེས་པ་ནི", "ཞེས་གསུངས", "ཅེས་གསུངས",
                  "ཞེས་བྱ་བ", "ཞེས་པའོ", "ཞེས་པ", "ཅེས་པ", "ཞེས་", "ཅེས་"],
                 key=len, reverse=True)


def transition_matrix(bp: float) -> np.ndarray:
    m = np.zeros((3, 3))
    for p in range(3):
        for q in range(3):
            if q == I and p not in (B, I):
                m[p, q] = NEG
                continue
            if p in (B, I) and q != I:
                m[p, q] -= bp
    return m


def viterbi(lg: np.ndarray, bp: float) -> np.ndarray:
    T, C = lg.shape
    tr = transition_matrix(bp)
    dp = np.full((T, C), NEG)
    back = np.zeros((T, C), dtype=np.int64)
    dp[0] = lg[0]
    dp[0, I] = NEG
    for t in range(1, T):
        s = dp[t - 1][:, None] + tr
        back[t] = s.argmax(0)
        dp[t] = s.max(0) + lg[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def spans(seq) -> list[tuple[int, int]]:
    out, start = [], None
    for k, v in enumerate(seq):
        if v == B:
            if start is not None:
                out.append((start, k - 1))
            start = k
        elif v == I:
            if start is None:
                start = k
        else:
            if start is not None:
                out.append((start, k - 1)); start = None
    if start is not None:
        out.append((start, len(seq) - 1))
    return out


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    return inter / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)


def matches(gold, pred, thr=0.5):
    c = sorted(((iou(g, p), gi, pi) for pi, p in enumerate(pred)
                for gi, g in enumerate(gold) if iou(g, p) >= thr), reverse=True)
    ug, up, n = set(), set(), 0
    for _, gi, pi in c:
        if gi in ug or pi in up:
            continue
        ug.add(gi); up.add(pi); n += 1
    return n


def particle_mask(ids, tok) -> np.ndarray:
    """Boolean per token: does this token overlap a quote particle?"""
    pieces = [tok.decode([i], skip_special_tokens=True).replace("▁", "")
              for i in ids]
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos)
        pos += len(p)
    text = "".join(pieces)
    mask = np.zeros(len(ids), dtype=bool)
    covered = np.zeros(len(text), dtype=bool)
    for c in CLOSERS:
        k = text.find(c)
        while k != -1:
            if not covered[k:k + len(c)].any():
                covered[k:k + len(c)] = True
            k = text.find(c, k + 1)
    ends = starts[1:] + [len(text)]
    for t, (s, e) in enumerate(zip(starts, ends)):
        if e > s and covered[s:e].any():
            mask[t] = True
    return mask


def score(gold_all, pred_all):
    TP = FP = FN = 0
    for g, p in zip(gold_all, pred_all):
        m = matches(g, p)
        TP += m; FP += len(p) - m; FN += len(g) - m
    pr = TP / (TP + FP) if TP + FP else 0.0
    rc = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return f1, pr, rc, TP + FP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Yontenn/mmbert-tsawa-bio-inv-v2")
    ap.add_argument("--dataset", default="ds_v2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--break-penalty", type=float, default=12.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ds = (load_from_disk(args.dataset) if Path(args.dataset).exists()
          else load_dataset(args.dataset, token=os.environ.get("HF_TOKEN")))
    split = ds[args.split]
    if args.limit:
        split = split.select(range(args.limit))
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval().to(args.device)
    if model.config.num_labels != 3:
        raise SystemExit("this expects a 3-label BIO model")

    keep = ["input_ids", "attention_mask", "labels"]
    t = split.remove_columns([c for c in split.column_names if c not in keep])
    t.set_format("torch")
    print(f"{args.split}: {len(t)} windows, device {args.device}")

    gold_all, base_all, rule_all = [], [], []
    fired = total_tok = 0
    with torch.no_grad():
        for i in range(0, len(t), args.batch_size):
            b = t[i:i + args.batch_size]
            lg = model(input_ids=b["input_ids"].to(args.device),
                       attention_mask=b["attention_mask"].to(args.device)
                       ).logits.float().cpu().numpy()
            for j in range(len(lg)):
                m = b["labels"][j].numpy() != -100
                ids = b["input_ids"][j].numpy()[m]
                l = lg[j][m]
                gold_all.append(spans(b["labels"][j].numpy()[m]))
                base_all.append(spans(viterbi(l, args.break_penalty)))

                pm = particle_mask(ids, tok)
                fired += int(pm.sum()); total_tok += len(pm)
                l2 = l.copy()
                l2[pm, B] = NEG
                l2[pm, I] = NEG
                rule_all.append(spans(viterbi(l2, args.break_penalty)))
            if (i // args.batch_size) % 25 == 0:
                print(f"  {i}/{len(t)}", flush=True)

    b_f1, b_p, b_r, b_n = score(gold_all, base_all)
    r_f1, r_p, r_r, r_n = score(gold_all, rule_all)
    n_gold = sum(len(g) for g in gold_all)

    print(f"\n{'='*62}")
    print(f"PARTICLE RULE - {args.model}")
    print("=" * 62)
    print(f"  tokens masked as particles: {fired:,} of {total_tok:,} "
          f"({100*fired/max(total_tok,1):.2f}%)")
    print(f"  gold spans: {n_gold:,}")
    print(f"\n  {'':<18}{'IoU@0.5':>9}{'P':>9}{'R':>9}{'pred':>8}")
    print(f"  {'Viterbi':<18}{b_f1:>9.4f}{b_p:>9.4f}{b_r:>9.4f}{b_n:>8,}")
    print(f"  {'Viterbi + rule':<18}{r_f1:>9.4f}{r_p:>9.4f}{r_r:>9.4f}{r_n:>8,}")
    print(f"  {'change':<18}{r_f1-b_f1:>+9.4f}{r_p-b_p:>+9.4f}{r_r-b_r:>+9.4f}")
    print()
    if r_f1 - b_f1 > 0.01:
        print("  The rule helps. Overruns across particles were costing matches.")
    elif r_f1 - b_f1 < -0.01:
        print("  The rule hurts - some gold spans do contain particles, or the")
        print("  token matching is catching particles inside real root text.")
    else:
        print("  No real effect. Overruns mostly don't cross a particle, so the")
        print("  containment failures happen somewhere the rule can't reach.")


if __name__ == "__main__":
    main()