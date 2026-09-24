#!/usr/bin/env python3
"""
Build the Sabche BIO DatasetDict for mmBERT (no training).

Usage:
    python src/build_sabche_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from build_tsawa_dataset import (  # noqa: E402
    IGNORE_LABEL,
    label_tokens,
    pack_window,
    sliding_windows,
    special_token_ids,
)

LABEL2ID = {"O": 0, "B-SABCHE": 1, "I-SABCHE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audit-csv", type=Path, default=ROOT / "data/tsawa_audit.csv")
    p.add_argument("--sidecar", type=Path, default=ROOT / "data/sabche_spans_clean.csv")
    p.add_argument("--split-file", type=Path, default=ROOT / "data/split_frozen.csv")
    p.add_argument("--tokenizer", default="jhu-clsp/mmBERT-base")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--stride", type=int, default=5120)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/sabche_dataset")
    p.add_argument("--stats-json", type=Path, default=ROOT / "data/sabche_dataset_stats.json")
    p.add_argument("--limit", type=int, default=None, help="debug: first N documents only")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from datasets import Dataset, DatasetDict, Features, Sequence, Value
    from transformers import AutoTokenizer

    audit = {r["pecha_id"]: r for r in csv.DictReader(args.audit_csv.open(encoding="utf-8"))}
    split = {r["pecha_id"]: r["split"] for r in csv.DictReader(
        l for l in args.split_file.open(encoding="utf-8") if not l.startswith("#"))}
    spans = defaultdict(list)
    for r in csv.DictReader(args.sidecar.open(encoding="utf-8")):
        if r["dropped"] == "True" or r["pecha_id"] not in split:
            continue
        spans[r["pecha_id"]].append((int(r["start"]), int(r["end"]), r["ann_id"], "SABCHE"))
    ids = sorted(split)
    if args.limit:
        ids = ids[: args.limit]
    missing = [p for p in ids if p not in spans]
    if missing:
        raise SystemExit(f"split lists books with no active spans: {missing[:10]}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tok.is_fast:
        raise SystemExit("A fast tokenizer is required for offset mapping.")
    tok.model_max_length = int(1e12)
    cls_id, sep_id, pad_id = special_token_ids(tok)
    content_len = args.max_length - 2

    out = {s: [] for s in SPLIT_NAMES.values()}
    counts = {s: Counter() for s in SPLIT_NAMES.values()}
    doc_counts = {s: Counter() for s in SPLIT_NAMES.values()}
    lost = Counter()
    for n, pid in enumerate(ids):
        text = Path(audit[pid]["base_path"]).read_text(encoding="utf-8")
        sp = sorted(spans[pid])
        # spans must not overlap for label_tokens; audit found none, assert it
        for (s1, e1, *_), (s2, *_) in zip(sp, sp[1:]):
            if s2 < e1:
                raise SystemExit(f"{pid}: overlapping spans at {s1}-{e1} / {s2}")
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = [tuple(o) for o in enc["offset_mapping"]]
        labels = label_tokens(offs, sp, LABEL2ID)
        n_b = sum(1 for x in labels if x == LABEL2ID["B-SABCHE"])
        lost["spans"] += len(sp)
        lost["spans_without_token"] += len(sp) - n_b
        sname = SPLIT_NAMES[split[pid]]
        doc_counts[sname]["docs"] += 1
        doc_counts[sname]["spans"] += len(sp)
        doc_counts[sname][pid[0]] += 1
        cov = 100.0 * sum(x[1] - x[0] for x in sp) / max(len(text), 1)
        for wi, (ws, we) in enumerate(sliding_windows(len(enc["input_ids"]), content_len, args.stride)):
            ex = pack_window(enc["input_ids"], labels, offs, ws, we, cls_id, sep_id, pad_id,
                             args.max_length)
            ex.update({"pecha_id": pid, "source_batch": "old" if pid.startswith("P") else "new",
                       "window_index": wi, "n_tokens_doc": len(enc["input_ids"]),
                       "coverage_pct": round(cov, 4)})
            counts[sname].update(x for x in ex["labels"] if x != IGNORE_LABEL)
            out[sname].append(ex)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(ids)} docs", flush=True)

    feats = Features({
        "input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(Value("int32")), "token_start": Value("int32"),
        "token_end": Value("int32"), "char_start": Value("int32"), "char_end": Value("int32"),
        "pecha_id": Value("string"), "source_batch": Value("string"),
        "window_index": Value("int32"), "n_tokens_doc": Value("int32"),
        "coverage_pct": Value("float64"),
    })
    dd = DatasetDict({
        s: Dataset.from_list(v, features=feats) if v
        else Dataset.from_dict({k: [] for k in feats}, features=feats)
        for s, v in out.items()
    })
    dd.save_to_disk(str(args.out_dir))

    # inverse-frequency class weights from TRAIN only (sum(n)/(K*n_c)), windows
    # overlap so this counts each token once per window it appears in
    tr = counts["train"]
    total = sum(tr.values())
    weights = {ID2LABEL[c]: round(total / (len(LABEL2ID) * tr[c]), 4) for c in sorted(tr)}
    stats = {
        "label2id": LABEL2ID, "max_length": args.max_length, "stride": args.stride,
        "tokenizer": args.tokenizer,
        "windows": {s: len(v) for s, v in out.items()},
        "docs": {s: dict(c) for s, c in doc_counts.items()},
        "label_counts": {s: {ID2LABEL[k]: v for k, v in sorted(c.items())} for s, c in counts.items()},
        "class_weights_inverse_freq_train": weights,
        "spans_total": lost["spans"], "spans_without_B_token": lost["spans_without_token"],
    }
    args.stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"saved {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
