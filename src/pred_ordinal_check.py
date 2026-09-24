#!/usr/bin/env python3
"""
pred_ordinal_check.py - do predicted Sabche spans start on an ordinal as
often as gold spans do? Self-contained: meant for the Vast GPU box.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

NEG = -1.0e9
O, B, I = 0, 1, 2
ORD = ("དང་པོ|གཉིས་པ|གསུམ་པ|བཞི་པ|ལྔ་པ|དྲུག་པ|བདུན་པ|བརྒྱད་པ|དགུ་པ|བཅུ་པ|"
       "བཅུ་(?:གཅིག|གཉིས|གསུམ|བཞི|ལྔ|དྲུག|བདུན|བརྒྱད|དགུ)་པ|བཅོ་(?:ལྔ|བརྒྱད)་པ|ཉི་ཤུ་པ")
START = re.compile(r"^[\s༈]*(?:[\(\{\[]?[0-9༠-༩]+[\)\}\]༽\.]?\s*)?(?:ད་ནི་|དེ་ནས་)?(?:" + ORD + ")")


# decoder + metric: copied from src/eval_viterbi_iou.py
def transition_matrix(break_penalty: float) -> np.ndarray:
    labels = ["O", "B", "I"]
    m = np.zeros((3, 3), dtype=np.float64)
    for i, prev in enumerate(labels):
        for j, nxt in enumerate(labels):
            if nxt == "I" and prev not in ("B", "I"):
                m[i, j] = NEG
                continue
            if prev == "O":
                continue
            if nxt != "I":
                m[i, j] -= break_penalty
    return m


def viterbi(logits: np.ndarray, break_penalty: float) -> np.ndarray:
    T, C = logits.shape
    trans = transition_matrix(break_penalty)
    dp = np.full((T, C), NEG)
    bp = np.zeros((T, C), dtype=np.int64)
    dp[0] = logits[0]
    dp[0, I] = NEG
    for t in range(1, T):
        scores = dp[t - 1][:, None] + trans
        bp[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + logits[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def spans_from_bio(seq) -> list[tuple[int, int]]:
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


def inclusive_iou(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union else 0.0


def match_iou(gold, pred, threshold: float):
    cands = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            if p[0] > g[1] or g[0] > p[1]:
                continue
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



def load_split(dataset: str, split: str):
    from datasets import Dataset

    p = Path(dataset)
    if not p.exists():  # Hub repo saved with save_to_disk
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(dataset, repo_type="dataset",
                                   allow_patterns=[f"{split}/*"],
                                   token=os.environ.get("HF_TOKEN")))
    return Dataset.load_from_disk(str(p / split))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Yontenn/mmbert-sabche-v1")
    ap.add_argument("--dataset", default="Yontenn/formatting-sabche-v1",
                    help="Hub dataset id or local save_to_disk folder")
    ap.add_argument("--tokenizer", default="jhu-clsp/mmBERT-base",
                    help="tokenizer the dataset was built with (model repo has none)")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--penalties", default="2.0,4.0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-dir", default="sabche_ordinal_check")
    args = ap.parse_args()
    if args.split == "test":
        print("note: test is frozen; report it, don't tune on it")
    pens = [float(x) for x in args.penalties.split(",")]

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_split(args.dataset, args.split)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).eval().to(dev)
    print(f"{args.split}: {len(ds)} windows on {dev}; labels {model.config.id2label}")

    # per book: token -> (distance to window centre, id, gold label, pred label per pen)
    best = defaultdict(dict)
    for i in range(0, len(ds), args.batch_size):
        batch = ds[i: i + args.batch_size]
        ids = torch.tensor(batch["input_ids"], device=dev)
        am = torch.tensor(batch["attention_mask"], device=dev)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=am).logits.float().cpu().numpy()
        for j in range(len(ids)):
            lab = np.array(batch["labels"][j])
            keep = np.nonzero(lab != -100)[0]  # content tokens (after CLS)
            lg = logits[j][keep]
            t0, t1 = batch["token_start"][j], batch["token_end"][j]
            centre = (t0 + t1) / 2
            pid = batch["pecha_id"][j]
            preds = {p: viterbi(lg, p) for p in pens}
            store = best[pid]
            for k, pos in enumerate(keep):
                ti = t0 + k
                d = abs(ti - centre)
                if ti not in store or d < store[ti][0]:
                    store[ti] = (d, int(batch["input_ids"][j][pos]), int(lab[pos]),
                                 {p: int(preds[p][k]) for p in pens})
        if (i // args.batch_size) % 25 == 0:
            print(f"  {i}/{len(ds)} windows", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for pen in pens:
        c = Counter()
        rows = []
        for pid in sorted(best):
            store = best[pid]
            n = max(store) + 1
            tok_ids = [store[t][1] if t in store else tok.pad_token_id for t in range(n)]
            gold = spans_from_bio([store[t][2] if t in store else O for t in range(n)])
            pred = spans_from_bio([store[t][3][pen] if t in store else O for t in range(n)])
            m = match_iou(gold, pred, 0.5)
            mg, mp = {a for a, _, _ in m}, {b for _, b, _ in m}

            def text(sp):
                return tok.decode(tok_ids[sp[0]: sp[1] + 1]).strip()

            for k, g in enumerate(gold):
                t = text(g)
                o = bool(START.match(t))
                hit = k in mg
                c["gold"] += 1
                c["gold_ord"] += o
                c["hit" if hit else "miss"] += 1
                c["hit_ord" if hit else "miss_ord"] += o
                if not hit:
                    rows.append({"pecha_id": pid, "kind": "missed_gold", "tok_start": g[0],
                                 "tok_end": g[1], "ordinal_start": o, "text": t[:120]})
            for k, p in enumerate(pred):
                t = text(p)
                o = bool(START.match(t))
                c["pred"] += 1
                c["pred_ord"] += o
                if k not in mp:
                    c["fp"] += 1
                    c["fp_ord"] += o
                    rows.append({"pecha_id": pid, "kind": "false_pred", "tok_start": p[0],
                                 "tok_end": p[1], "ordinal_start": o, "text": t[:120]})
        tp = c["hit"]
        prec = tp / c["pred"] if c["pred"] else 0.0
        rec = tp / c["gold"] if c["gold"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

        def pct(a, b):
            return round(100 * c[a] / c[b], 1) if c[b] else None

        rep = {
            "split": args.split, "break_penalty": pen, "books": len(best),
            "book_level_iou50": {"f1": round(f1, 4), "precision": round(prec, 4),
                                 "recall": round(rec, 4), "tp": tp,
                                 "n_gold": c["gold"], "n_pred": c["pred"]},
            "ordinal_start_pct": {"gold": pct("gold_ord", "gold"),
                                  "predicted": pct("pred_ord", "pred"),
                                  "matched_gold": pct("hit_ord", "hit"),
                                  "missed_gold": pct("miss_ord", "miss"),
                                  "false_predictions": pct("fp_ord", "fp")},
            "counts": dict(c),
        }
        report[str(pen)] = rep
        with (out / f"{args.split}_errors_bp{pen}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["pecha_id", "kind", "tok_start", "tok_end",
                                               "ordinal_start", "text"])
            w.writeheader()
            w.writerows(rows)
        print(json.dumps(rep, indent=1, ensure_ascii=False))
    (out / f"{args.split}_ordinal_check.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
