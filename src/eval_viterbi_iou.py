#!/usr/bin/env python3
"""
eval_viterbi_iou.py - score a tsawa model the way the rest of the team does.

Usage
-----
    python eval_viterbi_iou.py --model <path-or-hub-id> --dataset <path>

    # find the best break_penalty on validation
    python eval_viterbi_iou.py --model ... --sweep

Works on BIO models (3 labels). For an IO model (2 labels) Viterbi has almost
nothing to constrain - there are no illegal transitions - so the script says
so and falls back to argmax.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForTokenClassification

NEG = -1.0e9
O, B, I = 0, 1, 2


# decoding

def transition_matrix(break_penalty: float) -> np.ndarray:
    """
    3-label BIO transitions. I may only follow B or I; every span exit costs
    `break_penalty`, so fragmenting a span is penalised.
    """
    labels = ["O", "B", "I"]
    n = len(labels)
    m = np.zeros((n, n), dtype=np.float64)
    for i, prev in enumerate(labels):
        for j, nxt in enumerate(labels):
            if nxt == "I" and prev not in ("B", "I"):
                m[i, j] = NEG
                continue
            if prev == "O":
                continue
            # leaving a span (B/I -> anything that is not I) costs the penalty
            if nxt != "I":
                m[i, j] -= break_penalty
    return m


def viterbi(logits: np.ndarray, break_penalty: float) -> np.ndarray:
    """logits [T, C] -> best legal label sequence [T]."""
    T, C = logits.shape
    trans = transition_matrix(break_penalty)
    dp = np.full((T, C), NEG)
    bp = np.zeros((T, C), dtype=np.int64)
    dp[0] = logits[0]
    dp[0, I] = NEG  # a window cannot open mid-span
    for t in range(1, T):
        scores = dp[t - 1][:, None] + trans  # [C_prev, C_next]
        bp[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + logits[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def spans_from_bio(seq: np.ndarray) -> list[tuple[int, int]]:
    """BIO -> inclusive (start, end) spans, matching the team's convention."""
    out, start = [], None
    for i, v in enumerate(seq):
        if v == B:
            if start is not None:
                out.append((start, i - 1))
            start = i
        elif v == I:
            if start is None:
                start = i
        else:
            if start is not None:
                out.append((start, i - 1))
                start = None
    if start is not None:
        out.append((start, len(seq) - 1))
    return out


# metric - mirrors v2_metrics.inclusive_iou / match_iou

def inclusive_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union else 0.0


def match_iou(gold, pred, threshold: float):
    """Greedy best-first one-to-one, exactly as the team repo does it."""
    cands = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            v = inclusive_iou(g, p)
            if v >= threshold:
                cands.append((v, gi, pi))
    cands.sort(reverse=True)
    used_g, used_p, matches = set(), set(), []
    for v, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, v))
    return matches


def score(all_gold, all_pred, threshold: float):
    TP = FP = FN = 0
    for g, p in zip(all_gold, all_pred):
        m = match_iou(g, p, threshold)
        TP += len(m)
        FP += len(p) - len(m)
        FN += len(g) - len(m)
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": f1, "precision": prec, "recall": rec,
            "tp": TP, "fp": FP, "fn": FN,
            "n_gold": TP + FN, "n_pred": TP + FP,
            "overproposal": (TP + FP) / (TP + FN) if TP + FN else 0.0}



def dump_spans(path, meta, kept_pos, gold_all, preds):
    """
    One JSON line per window, spans inclusive. The scored sequence drops -100
    positions (CLS/SEP/pad and ignore spans); kept_pos maps each scored index
    back to its window position, and window position 1 (after CLS) is book
    token ``token_start``.
    """
    def to_book(spans, pos, t0):
        return [[int(t0 + pos[a] - 1), int(t0 + pos[b] - 1)] for a, b in spans]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for k, (mt, pos) in enumerate(zip(meta, kept_pos)):
            t0 = int(mt["token_start"])
            # "<name>" = scored-index spans (what score() matches on);
            # "<name>_tok" = the same spans as book-token indices
            row = {**mt, "n_scored": int(len(pos)),
                   "gold": [list(map(int, x)) for x in gold_all[k]],
                   "gold_tok": to_book(gold_all[k], pos, t0)}
            for name, pr in preds.items():
                row[name] = [list(map(int, x)) for x in pr[k]]
                row[f"{name}_tok"] = to_book(pr[k], pos, t0)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="Yontenn/formatting-tsawa-v6")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--break-penalty", type=float, default=2.0,
                    help="team default is 2.0")
    ap.add_argument("--sweep", action="store_true",
                    help="try several break penalties and report each")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/tsawa_viterbi_eval.json")
    ap.add_argument("--dump-spans", default="",
                    help="also write per-window gold/pred spans as JSONL, in "
                         "book-token coordinates (see rescore_masked.py)")
    args = ap.parse_args()

    ds = (load_from_disk(args.dataset) if Path(args.dataset).exists()
          else load_dataset(args.dataset, token=os.environ.get("HF_TOKEN")))
    split = ds[args.split]
    if args.limit:
        split = split.select(range(args.limit))
    print(f"{args.split}: {len(split)} windows")

    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval().to(args.device)
    n_lab = model.config.num_labels
    print(f"model: {n_lab} labels ({'BIO' if n_lab == 3 else 'IO'})")
    if n_lab != 3:
        print("\n  NOTE: Viterbi constrains BIO transitions (I may not follow O).")
        print("  With 2 labels there are no illegal transitions, so decoding")
        print("  reduces to argmax and only the metric below differs.\n")

    meta_cols = ["pecha_id", "window_index", "token_start", "char_start", "char_end"]
    meta = ([dict(zip(meta_cols, row)) for row in zip(*(split[c] for c in meta_cols))]
            if args.dump_spans else [])

    keep = ["input_ids", "attention_mask", "labels"]
    t = split.remove_columns([c for c in split.column_names if c not in keep])
    t.set_format("torch")

    # collect logits and gold once; decoding is cheap to repeat
    logits_all, gold_all, kept_pos = [], [], []
    with torch.no_grad():
        for i in range(0, len(t), args.batch_size):
            b = t[i: i + args.batch_size]
            out = model(input_ids=b["input_ids"].to(args.device),
                        attention_mask=b["attention_mask"].to(args.device)).logits
            lg = out.float().cpu().numpy()
            lab = b["labels"].numpy()
            for j in range(len(lab)):
                m = lab[j] != -100
                logits_all.append(lg[j][m])
                if args.dump_spans:
                    kept_pos.append(np.nonzero(m)[0])
                gold_all.append(spans_from_bio(lab[j][m]))
            if (i // args.batch_size) % 25 == 0:
                print(f"  {i}/{len(t)}", flush=True)

    penalties = ([0.0, 0.5, 1.0, 2.0, 3.0, 5.0] if args.sweep
                 else [args.break_penalty])
    results = {}

    # argmax baseline, for the comparison that matters
    argmax_pred = [spans_from_bio(lg.argmax(-1)) for lg in logits_all]
    base = score(gold_all, argmax_pred, 0.5)
    print(f"\n{'='*64}")
    print("ARGMAX (what you were doing)")
    print("=" * 64)
    print(f"  IoU@0.5  F1={base['f1']:.4f}  P={base['precision']:.4f} "
          f"R={base['recall']:.4f}  pred/gold={base['overproposal']:.2f}x")
    results["argmax"] = base

    print(f"\n{'='*64}")
    print("VITERBI")
    print("=" * 64)
    for bp in penalties:
        pred = [spans_from_bio(viterbi(lg, bp)) for lg in logits_all]
        r = score(gold_all, pred, 0.5)
        results[f"viterbi_bp{bp}"] = r
        delta = r["f1"] - base["f1"]
        print(f"  break_penalty={bp:<4} IoU@0.5 F1={r['f1']:.4f} "
              f"({delta:+.4f})  P={r['precision']:.4f} R={r['recall']:.4f} "
              f"pred/gold={r['overproposal']:.2f}x")

    # thresholds at the chosen penalty
    bp = args.break_penalty if not args.sweep else max(
        penalties, key=lambda x: results[f"viterbi_bp{x}"]["f1"])
    pred = [spans_from_bio(viterbi(lg, bp)) for lg in logits_all]
    print(f"\n{'='*64}")
    print(f"THRESHOLD SWEEP at break_penalty={bp}")
    print("=" * 64)
    for thr in (0.5, 0.7, 0.9, 1.0):
        r = score(gold_all, pred, thr)
        results[f"iou{int(thr*100)}"] = r
        print(f"  IoU>={thr:.1f}  F1={r['f1']:.4f}  P={r['precision']:.4f} "
              f"R={r['recall']:.4f}")

    print("\n  Team reference: yigchung single-layer model reports "
          "iou50_f1 = 0.389")
    print("  (different layer, same metric and decoder - the closest "
          "comparable number.)")

    if args.dump_spans:
        dump_spans(args.dump_spans, meta, kept_pos, gold_all,
                   {"argmax": argmax_pred, f"viterbi_bp{bp}": pred})
        print(f"per-window spans written to {args.dump_spans}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()