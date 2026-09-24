#!/usr/bin/env python3
"""
score_spans.py - score predicted character spans against the Sabche v2 gold.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def iou(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    return inter / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)


def score(gold, pred, thr: float = 0.5) -> dict:
    cands = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gold)
                    for pi, p in enumerate(pred) if iou(g, p) >= thr), reverse=True)
    ug, up = set(), set()
    for _v, gi, pi in cands:
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
    return {"tp": len(ug), "n_pred": len(pred), "n_gold": len(gold)}


def prf(tp: int, n_pred: int, n_gold: int):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", action="append", required=True, metavar="NAME=SPAN_DIR")
    ap.add_argument("--per-book", action="store_true")
    ap.add_argument("--json", default="", help="also write all numbers to this file")
    args = ap.parse_args()

    manifest = {r["pecha_id"]: r for r in csv.DictReader(
        open(ROOT / "data/books_manifest.csv", encoding="utf-8"))}
    books = sorted(b for b, r in manifest.items() if r["split"] == args.split)
    gold = {b: [] for b in books}
    for r in csv.DictReader(open(ROOT / "data/sabche_gold.csv", encoding="utf-8")):
        if r["pecha_id"] in gold:
            gold[r["pecha_id"]].append((int(r["start"]), int(r["end_exclusive"]) - 1))

    per: dict[str, dict[str, dict]] = {}
    for spec in args.model:
        name, d = spec.split("=", 1)
        per[name] = {}
        for b in books:
            f = Path(d) / f"{b}.json"
            if not f.is_file():
                raise SystemExit(f"{name}: missing {f}")
            pred = [tuple(x) for x in json.loads(f.read_text())["spans"]]
            per[name][b] = score(sorted(gold[b]), pred)

    groups = [
        (f"all {len(books)} books", books),
        ("old-batch", [b for b in books if manifest[b]["source_batch"] == "old"]),
        ("new-batch", [b for b in books if manifest[b]["source_batch"] == "new"]),
    ]
    names = list(per)
    width = 26
    print(f"split={args.split}")
    print(f"{'':16}" + "".join(f"{n:<{width}}" for n in names))
    out = {}
    for label, bs in groups:
        cells = []
        for n in names:
            tp = sum(per[n][b]["tp"] for b in bs)
            npd = sum(per[n][b]["n_pred"] for b in bs)
            ng = sum(per[n][b]["n_gold"] for b in bs)
            f1, p, r = prf(tp, npd, ng)
            out.setdefault(n, {})[label] = {"f1": f1, "precision": p, "recall": r,
                                            "tp": tp, "n_pred": npd, "n_gold": ng}
            cells.append(f"F1 {f1:.3f} P {p:.3f} R {r:.3f}")
        print(f"{label:16}" + "".join(f"{c:<{width}}" for c in cells))
    med = []
    for n in names:
        m = statistics.median(prf(per[n][b]["tp"], per[n][b]["n_pred"], per[n][b]["n_gold"])[0]
                              for b in books)
        out[n]["median_book_f1"] = m
        med.append(f"F1 {m:.3f}")
    print(f"{'median book':16}" + "".join(f"{c:<{width}}" for c in med))

    if args.per_book:
        print(f"\n{'book':11}{'batch':6}{'gold':>6}  " + "  ".join(f"{n[:22] + ' F1/FP':>26}" for n in names))
        first = names[0]
        order = sorted(books, key=lambda b: -per[first][b]["n_pred"] + per[first][b]["tp"])
        for b in order:
            cells = []
            for n in names:
                s = per[n][b]
                cells.append(f"{prf(s['tp'], s['n_pred'], s['n_gold'])[0]:>18.3f} {s['n_pred'] - s['tp']:>7d}")
            print(f"{b:11}{manifest[b]['source_batch']:6}{per[first][b]['n_gold']:>6}  " + "  ".join(cells))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
