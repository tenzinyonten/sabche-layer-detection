#!/usr/bin/env python3
"""
mmbert_predict.py - run a trained mmBERT layer model (tsawa, sabche or chapter) over whole
books and write character-offset spans, with no tokenized dataset needed.

Each book is tokenized once and cut into 8,190-token windows that start every 5,120 tokens, the same
windows the datasets use. Every window is decoded with Viterbi, the spans are mapped back to
characters, and spans found twice in the overlap are removed. Output: <out>/spans/<book>.json =
{"book", "spans": [[start, end]]} with inclusive character offsets, ready for src/score_spans.py.

    python src/mmbert_predict.py --model Yontenn/mmbert-chapter-v1 --texts-dir data/raw_opf \\
        --books I319DAFF7 --out results/mmbert-chapter-v1/test --break-penalty 4.0

A window of 8,192 tokens is heavy on a CPU (tens of seconds each), so use a GPU for whole splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decode import spans_from_bio, viterbi  # noqa: E402

MAX_LENGTH = 8192
STRIDE = 5120


def sliding_windows(n_tokens: int, content_len: int, stride: int) -> list[tuple[int, int]]:
    """Token windows covering [0, n_tokens); the last window is flush-right."""
    if n_tokens <= 0:
        return []
    if n_tokens <= content_len:
        return [(0, n_tokens)]
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + content_len, n_tokens)
        spans.append((start, end))
        if end == n_tokens:
            break
        start += stride
        if start >= n_tokens:
            break
        if start + content_len >= n_tokens and end < n_tokens:
            spans.append((max(0, n_tokens - content_len), n_tokens))
            break
    out, seen = [], set()
    for w in spans:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def iou(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    return inter / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)


def dedupe(spans, thr: float = 0.5):
    """Overlapping windows can find the same span twice; keep the first."""
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


@torch.no_grad()
def predict_book(model, tok, text: str, break_penalty: float, device: str):
    enc = tok(text, add_special_tokens=False, truncation=False,
              return_offsets_mapping=True, return_attention_mask=False)
    ids = list(enc["input_ids"])
    offs = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    content_len = MAX_LENGTH - 2
    spans = []
    windows = sliding_windows(len(ids), content_len, STRIDE)
    for wi, (a, b) in enumerate(windows):
        x = torch.tensor([[cls_id] + ids[a:b] + [sep_id]], device=device)
        logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0]
        content = logits[1:-1].float().cpu().numpy()            # drop CLS / SEP
        for ts, te in spans_from_bio(viterbi(content, break_penalty)):
            gs, ge = a + ts, a + te                              # book-token indices
            spans.append((offs[gs][0], offs[ge][1] - 1))         # inclusive chars
        print(f"    window {wi + 1}/{len(windows)}", flush=True)
    return dedupe(spans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--texts-dir", default="data/raw_opf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--break-penalty", type=float, default=4.0)
    ap.add_argument("--tokenizer", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if len(labels) != 3 or labels[0] != "O" or not labels[1].startswith("B-") or not labels[2].startswith("I-"):
        raise SystemExit(f"expected a 3-label BIO model (O, B-x, I-x), got {labels}")
    print(f"labels: {labels}")
    model.eval().to(args.device)

    outdir = Path(args.out) / "spans"
    outdir.mkdir(parents=True, exist_ok=True)
    for book in args.books:
        print(f"{book}", flush=True)
        text = load_book_text(Path(args.texts_dir), book)
        spans = predict_book(model, tok, text, args.break_penalty, args.device)
        (outdir / f"{book}.json").write_text(
            json.dumps({"book": book, "spans": spans}, ensure_ascii=False), encoding="utf-8")
        print(f"  {len(spans)} spans")


if __name__ == "__main__":
    main()
