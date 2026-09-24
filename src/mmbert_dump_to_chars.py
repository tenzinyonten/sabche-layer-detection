#!/usr/bin/env python3
"""
mmbert_dump_to_chars.py - turn a per-window token-span dump into per-book
character-offset spans.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def iou(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    return inter / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)


def dedupe(spans, thr: float = 0.5):
    kept = []
    for s in sorted(spans):
        if any(iou(s, k) >= thr for k in kept):
            continue
        kept.append(s)
    return kept


def load_book_text(texts_dir: Path, pecha_id: str) -> str:
    for rel in (f"{pecha_id}.opf/{pecha_id}.opf/base/v001.txt",
                f"{pecha_id}.opf/base/v001.txt"):
        p = texts_dir / rel
        if p.is_file():
            return p.read_text(encoding="utf-8")
    raise SystemExit(f"{pecha_id}: base/v001.txt not found under {texts_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--texts-dir", default="data/raw_opf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default=None,
                    help="prediction key ending in _tok (default: the first viterbi_bp*_tok)")
    ap.add_argument("--tokenizer", default="jhu-clsp/mmBERT-base")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dump, encoding="utf-8")]
    key = args.key or next(k for k in rows[0] if k.startswith("viterbi") and k.endswith("_tok"))
    print(f"prediction key: {key}")
    by_book = defaultdict(list)
    for r in rows:
        by_book[r["pecha_id"]].append(r)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    outdir = Path(args.out) / "spans"
    outdir.mkdir(parents=True, exist_ok=True)
    for book, wins in sorted(by_book.items()):
        text = load_book_text(Path(args.texts_dir), book)
        off = tok(text, add_special_tokens=False, truncation=False,
                  return_offsets_mapping=True)["offset_mapping"]
        for r in wins:   # the mapping must reproduce the dataset's own windows
            real = [o for o in off[r["token_start"]:] if o[1] > o[0]]
            if real[0][0] != r["char_start"]:
                raise SystemExit(f"{book} window {r['window_index']}: tokenization "
                                 f"does not reproduce char_start {r['char_start']}")
        spans = [(off[a][0], off[b][1] - 1) for r in wins for a, b in r[key]]
        spans = dedupe(spans)
        (outdir / f"{book}.json").write_text(
            json.dumps({"book": book, "spans": spans}, ensure_ascii=False), encoding="utf-8")
        print(f"{book}: {len(wins)} windows, {len(spans)} spans")


if __name__ == "__main__":
    main()
