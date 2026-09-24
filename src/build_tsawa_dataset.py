#!/usr/bin/env python3
"""
Build a binary tsawa BIO dataset from audited OpenPecha .opf repos.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from transformers import AutoTokenizer

# Frozen Phase 2 totals (2026-09-14 audit). Any drift → abort.
PHASE2_TOTAL_REPOS = 539
PHASE2_TSAWA_REPOS = 212
PHASE2_NEW_TSAWA = 123
PHASE2_OLD_TSAWA = 89
PHASE2_ZERO_LENGTH_SPANS = 41
# Frozen Phase-3 sidecar after snap + overlap resolve (non-zero YAML spans).
PHASE3_SIDECAR_ROWS = 21155
PHASE3_SIDECAR_DROPPED = 6
PHASE3_SIDECAR_ACTIVE = 21149
HIGH_DENSITY_OUTLIER = "IFE0B60AA"

LABEL2ID = {"O": 0, "B-TSAWA": 1, "I-TSAWA": 2}
ID2LABEL = {0: "O", 1: "B-TSAWA", 2: "I-TSAWA"}
# 5-label variant: adds the quotation class so the model can learn to reject
# citations from other works instead of scoring them as tsawa.
LABEL2ID_5 = {"O": 0, "B-TSAWA": 1, "I-TSAWA": 2, "B-QUOTE": 3, "I-QUOTE": 4}
ID2LABEL_5 = {v: k for k, v in LABEL2ID_5.items()}
IGNORE_LABEL = -100  # special tokens + padding (Trainer-compatible)

DEFAULT_TOKENIZER = "jhu-clsp/mmBERT-base"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a document-level-split tsawa BIO DatasetDict (no training)."
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw_opf"),
        help="Cloned .opf checkouts (default: data/raw_opf).",
    )
    p.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("data/tsawa_audit.csv"),
        help="Phase 2 audit CSV - source of truth for which repos to include.",
    )
    p.add_argument(
        "--source",
        choices=("combined", "new", "old"),
        default="combined",
        help="Which audit batch(es) to include (default: combined).",
    )
    p.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HF tokenizer name (default: {DEFAULT_TOKENIZER}).",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Window length in tokens, including special tokens (default: 8192).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=5120,
        help="Token stride between window starts (default: 5120).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the document-level split (default: 42).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/tsawa_dataset"),
        help="DatasetDict save_to_disk directory.",
    )
    p.add_argument(
        "--dropped-csv",
        type=Path,
        default=Path("data/dropped_spans.csv"),
        help="Log of spans excluded from labeling (zero-length YAML and/or resolver drops).",
    )
    p.add_argument(
        "--sidecar",
        type=Path,
        default=Path("data/tsawa_spans_resolved.csv"),
        help="Snapped+resolved span CSV (skip dropped=True). "
        "Pass empty string or use --from-yaml to label raw Tsawa.yml instead.",
    )
    p.add_argument(
        "--from-yaml",
        action="store_true",
        help="Ignore --sidecar and label from Tsawa.yml (pre-snap offsets).",
    )
    p.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Frozen split CSV (pecha_id,split[,group_id]); '#' lines are comments. "
        "Overrides the built-in coverage-quartile split. Splits: train/val|validation/test.",
    )
    p.add_argument(
        "--quote-sidecar",
        type=Path,
        default=None,
        help="Snapped Quotation/Citation span CSV. Switches the build to the "
        "5-label scheme (O/B-TSAWA/I-TSAWA/B-QUOTE/I-QUOTE). TSAWA wins overlaps.",
    )
    p.add_argument(
        "--exclude-pechas",
        default="",
        help="Comma-separated pecha IDs to drop from TRAIN. Refuses to drop a "
        "validation/test document, so eval sets stay frozen.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use at most N eligible repos (debug only).",
    )
    p.add_argument(
        "--restrict-to-split",
        action="store_true",
        help="Keep only pechas listed in --split-file (drops the rest from every split).",
    )
    p.add_argument(
        "--ignore-spans-csv",
        type=Path,
        default=None,
        help="CSV of pecha_id,start,end whose tokens are labeled -100.",
    )
    p.add_argument(
        "--add-features",
        action="store_true",
        help="Add a float [seq_len, 5] 'features' column from clause/syllable cues.",
    )
    return p.parse_args(argv)


def _as_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_and_validate_audit(path: Path) -> pd.DataFrame:
    """Load the Phase 2 CSV and abort if frozen totals no longer match."""
    if not path.is_file():
        raise SystemExit(f"Audit CSV not found: {path}\nRun Phase 2 first.")
    df = pd.read_csv(path)
    required = {
        "pecha_id",
        "source_batch",
        "has_tsawa_layer",
        "parse_status",
        "n_zero_length",
        "coverage_pct",
        "base_path",
        "tsawa_path",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Audit CSV missing columns {sorted(missing)}")

    n_total = len(df)
    has = _as_bool_series(df["has_tsawa_layer"])
    tsawa = df[has]
    n_tsawa = len(tsawa)
    n_new = int((tsawa["source_batch"] == "new").sum())
    n_old = int((tsawa["source_batch"] == "old").sum())
    n_zero = int(df["n_zero_length"].sum())

    problems: list[str] = []
    if n_total != PHASE2_TOTAL_REPOS:
        problems.append(f"total rows {n_total} != {PHASE2_TOTAL_REPOS}")
    if n_tsawa != PHASE2_TSAWA_REPOS:
        problems.append(f"has_tsawa_layer {n_tsawa} != {PHASE2_TSAWA_REPOS}")
    if n_new != PHASE2_NEW_TSAWA:
        problems.append(f"new-with-tsawa {n_new} != {PHASE2_NEW_TSAWA}")
    if n_old != PHASE2_OLD_TSAWA:
        problems.append(f"old-with-tsawa {n_old} != {PHASE2_OLD_TSAWA}")
    if n_zero != PHASE2_ZERO_LENGTH_SPANS:
        problems.append(f"zero-length span sum {n_zero} != {PHASE2_ZERO_LENGTH_SPANS}")
    bad_status = tsawa[tsawa["parse_status"] != "ok"]
    if len(bad_status):
        problems.append(
            f"{len(bad_status)} tsawa rows have parse_status != ok: "
            f"{bad_status['pecha_id'].tolist()[:8]}"
        )
    if problems:
        raise SystemExit(
            "Audit CSV does not match the frozen Phase 2 totals. "
            "Refusing to build a dataset from a drifted audit.\n  - "
            + "\n  - ".join(problems)
            + f"\nFile: {path}"
        )
    return df


def select_repos(audit: pd.DataFrame, source: str) -> pd.DataFrame:
    has = _as_bool_series(audit["has_tsawa_layer"])
    tsawa = audit.loc[has].copy()
    if source == "new":
        tsawa = tsawa[tsawa["source_batch"] == "new"]
    elif source == "old":
        tsawa = tsawa[tsawa["source_batch"] == "old"]
    return tsawa.reset_index(drop=True)


def load_cleaned_spans(
    tsawa_path: Path,
    pecha_id: str,
) -> tuple[list[tuple[int, int, str]], list[dict[str, Any]]]:
    """Return (kept half-open spans, dropped zero-length rows)."""
    data = yaml.safe_load(tsawa_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), dict):
        raise SystemExit(f"{pecha_id}: Tsawa.yml has no annotations mapping: {tsawa_path}")
    kept: list[tuple[int, int, str]] = []
    dropped: list[dict[str, Any]] = []
    for ann_id, payload in data["annotations"].items():
        if not isinstance(payload, dict):
            continue
        span = payload.get("span") or {}
        if not isinstance(span, dict):
            continue
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start == end:
            dropped.append(
                {
                    "pecha_id": pecha_id,
                    "annotation_id": str(ann_id),
                    "start": start,
                    "end": end,
                }
            )
            continue
        if start > end:
            # Audit said zero inverted spans; keep that invariant loud.
            raise SystemExit(
                f"{pecha_id}: inverted span {ann_id} start={start} end={end} "
                "(audit claimed n_inverted=0)"
            )
        kept.append((start, end, str(ann_id)))
    kept.sort(key=lambda t: (t[0], t[1]))
    return kept, dropped


def load_split_file(path: Path) -> tuple[dict[str, str], str]:
    """Read a frozen split CSV. Lines starting with '#' are metadata comments."""
    if not path.is_file():
        raise SystemExit(f"Split file not found: {path}")
    raw = path.read_text(encoding="utf-8").splitlines()
    comments = [ln for ln in raw if ln.startswith("#")]
    body = [ln for ln in raw if not ln.startswith("#")]
    reader = csv.DictReader(body)
    if not reader.fieldnames or "pecha_id" not in reader.fieldnames:
        raise SystemExit(f"{path}: needs a pecha_id column")
    if "split" not in reader.fieldnames:
        raise SystemExit(f"{path}: needs a split column")
    alias = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
    split_of: dict[str, str] = {}
    for row in reader:
        pid = (row["pecha_id"] or "").strip()
        if not pid:
            continue
        name = (row["split"] or "").strip().lower()
        if name not in alias:
            raise SystemExit(f"{path}: unknown split {name!r} for {pid}")
        if pid in split_of:
            raise SystemExit(f"{path}: {pid} listed twice")
        split_of[pid] = alias[name]
    if not split_of:
        raise SystemExit(f"{path}: no rows")
    seed_note = next(
        (ln.lstrip("# ").strip() for ln in comments if ln.lower().startswith("# seed")),
        "",
    )
    note = f"Frozen document split from `{path}`" + (f" ({seed_note})" if seed_note else "")
    return split_of, note


def load_sidecar_spans(path: Path) -> tuple[dict[str, list[tuple[int, int, str]]], list[dict[str, Any]]]:
    """Load snapped/resolved spans keyed by pecha_id. Skip dropped=True rows."""
    if not path.is_file():
        raise SystemExit(f"Span sidecar not found: {path}")
    by_repo: dict[str, list[tuple[int, int, str]]] = {}
    dropped: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pecha_id = row["pecha_id"]
            ann_id = row["ann_id"]
            start, end = int(row["start"]), int(row["end"])
            if str(row.get("dropped", "")).lower() == "true" or start >= end:
                dropped.append(
                    {
                        "pecha_id": pecha_id,
                        "annotation_id": ann_id,
                        "start": start,
                        "end": end,
                    }
                )
                continue
            by_repo.setdefault(pecha_id, []).append((start, end, ann_id))
    for pid in by_repo:
        by_repo[pid].sort(key=lambda t: (t[0], t[1]))
    return by_repo, dropped


def label_tokens(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int, str]],
    label2id: dict[str, int] | None = None,
) -> list[int]:
    """Assign O / B-<CLS> / I-<CLS> using the token-start rule.

    A token whose offset range straddles a span boundary is classified by
    ``token_start`` only: inside → B/I, outside → O. First token of each
    span (by token index) is B; later tokens whose start falls in the same
    span are I.

    ``spans`` are ``(start, end, ann_id)`` and default to class ``TSAWA``, or
    ``(start, end, ann_id, cls)`` for the multiclass build. Spans must be
    sorted by start and non-overlapping.
    """
    label2id = label2id or LABEL2ID
    labels = [label2id["O"]] * len(offsets)
    if not spans:
        return labels
    starts = [sp[0] for sp in spans]
    classes = [sp[3] if len(sp) > 3 else "TSAWA" for sp in spans]
    first_seen: set[int] = set()
    for i, (tok_s, tok_e) in enumerate(offsets):
        if tok_e <= tok_s:
            continue
        # Rightmost span with start <= tok_s (no overlaps in the audit).
        lo, hi = 0, len(starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] <= tok_s:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < 0:
            continue
        # Walk left in case of any unexpected overlap / equal starts.
        found = -1
        for j in range(idx, -1, -1):
            s, e = spans[j][0], spans[j][1]
            if e <= tok_s:
                continue
            if s <= tok_s < e:
                found = j
                break
            if s < tok_s:
                break
        if found < 0:
            continue
        cls = classes[found]
        if found not in first_seen:
            first_seen.add(found)
            labels[i] = label2id[f"B-{cls}"]
        else:
            labels[i] = label2id[f"I-{cls}"]
    return labels


def merge_intervals(pairs) -> list[tuple[int, int]]:
    """Sorted, coalesced [start, end) intervals."""
    out: list[tuple[int, int]] = []
    for s, e in sorted(pairs):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def in_intervals(intervals: list[tuple[int, int]], pos: int) -> bool:
    """Membership test against the output of ``merge_intervals``."""
    i = bisect_right(intervals, (pos, float("inf"))) - 1
    return i >= 0 and intervals[i][0] <= pos < intervals[i][1]


def apply_tsawa_precedence(
    tsawa: list[tuple[int, int, str]],
    quote: list[tuple[int, int, str]],
) -> tuple[list[tuple[int, int, str, str]], dict[str, int]]:
    """Merge tsawa and quotation spans into one non-overlapping sorted list.

    Precedence rule: **TSAWA wins every conflict**, because it is the target
    class and a token labeled QUOTE that is really root text is a false
    negative on the class we care about. Each quotation span is carved by the
    tsawa intervals it meets; the surviving pieces stay QUOTE. A quotation
    span cut in two by a tsawa span yields two pieces, and the second one
    opens a fresh ``B-QUOTE`` since it is a separate contiguous run.

    Returns the merged spans and a counter of what precedence cost.
    """
    tsawa = sorted(tsawa, key=lambda t: (t[0], t[1]))
    quote = sorted(quote, key=lambda t: (t[0], t[1]))
    stats = Counter()
    merged: list[tuple[int, int, str, str]] = [(s, e, a, "TSAWA") for s, e, a in tsawa]

    for qs, qe, qa in quote:
        stats["quote_spans_in"] += 1
        pieces = [(qs, qe)]
        for ts, te, _ta in tsawa:
            if te <= qs or ts >= qe:
                continue
            nxt = []
            for ps, pe in pieces:
                if te <= ps or ts >= pe:
                    nxt.append((ps, pe))
                    continue
                stats["chars_suppressed"] += min(pe, te) - max(ps, ts)
                if ps < ts:
                    nxt.append((ps, ts))
                if te < pe:
                    nxt.append((te, pe))
            pieces = nxt
        if len(pieces) == 1 and pieces[0] == (qs, qe):
            merged.append((qs, qe, qa, "QUOTE"))
            continue
        stats["quote_spans_touched"] += 1
        if not pieces:
            stats["quote_spans_dropped"] += 1
            continue
        if len(pieces) > 1:
            stats["quote_spans_fragmented"] += 1
        stats["quote_spans_clipped"] += 1
        for k, (ps, pe) in enumerate(sorted(pieces)):
            if pe > ps:
                merged.append((ps, pe, f"{qa}#{k}" if k else qa, "QUOTE"))

    merged.sort(key=lambda t: (t[0], t[1]))
    for a, b in zip(merged, merged[1:]):
        if b[0] < a[1]:
            raise SystemExit(f"precedence left an overlap: {a} vs {b}")
    stats["quote_spans_out"] = sum(1 for m in merged if m[3] == "QUOTE")
    return merged, stats


def sliding_windows(n_tokens: int, content_len: int, stride: int) -> list[tuple[int, int]]:
    """Token-index windows covering [0, n_tokens). Last window is flush-right."""
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
        # Guarantee the tail is covered even if stride skips past it.
        if start + content_len >= n_tokens and end < n_tokens:
            spans.append((max(0, n_tokens - content_len), n_tokens))
            break
    # Deduplicate if the flush-right window matches the previous one.
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in spans:
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def special_token_ids(tokenizer: Any) -> tuple[int, int, int]:
    cls_id = tokenizer.cls_token_id
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Gemma-style tokenizers sometimes leave pad unset.
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        pad_id = tokenizer.pad_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        raise SystemExit(
            f"Tokenizer is missing special ids (cls={cls_id}, sep={sep_id}, pad={pad_id})"
        )
    return int(cls_id), int(sep_id), int(pad_id)


CLAUSE_SEPS = set("།༎༏༐༑༔")
SYL_SEPS = set("་༌")
SOURCE_MARKERS = (
    "གཞུང་ལས",
    "རྒྱུད་ལས",
    "ལུང་ལས",
    "མདོ་ལས",
    "ལས།",
    "ནས།",
    "ལས་",
)
CLOSERS = ("ཞེས་པ", "ཅེས་པ", "ཞེས་", "ཅེས་", "གསུངས")
ZERO5 = (0.0, 0.0, 0.0, 0.0, 0.0)


def _n_syllables(clause: str) -> int:
    parts: list[str] = []
    buf: list[str] = []
    for ch in clause:
        if ch in SYL_SEPS:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return sum(1 for p in parts if p.strip())


def clause_features(text: str) -> tuple[list[tuple[int, int]], list[tuple[float, ...]]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in CLAUSE_SEPS:
            bounds.append((start, i + 1))
            start = i + 1
    if start < len(text) or not bounds:
        bounds.append((start, len(text)))
    n_syl = [_n_syllables(text[s:e]) for s, e in bounds]
    feats: list[tuple[float, float, float, float, float]] = []
    n = len(bounds)
    for i in range(n):
        f1 = min(n_syl[i] / 20.0, 1.0)
        f2 = 1.0 if n_syl[i] in (7, 9, 11, 13) else 0.0
        lo, hi = max(0, i - 2), min(n, i + 3)
        window = n_syl[lo:hi]
        std = float(np.std(window)) if len(window) > 1 else 0.0
        f3 = min(std / 10.0, 1.0)
        prev = text[bounds[i - 1][0]:bounds[i - 1][1]] if i else ""
        f4 = 1.0 if any(prev.endswith(m) for m in SOURCE_MARKERS) else 0.0
        nxt = "".join(text[bounds[j][0]:bounds[j][1]] for j in range(i + 1, min(n, i + 3)))
        f5 = 1.0 if any(c in nxt for c in CLOSERS) else 0.0
        feats.append((f1, f2, f3, f4, f5))
    return bounds, feats


def token_clause_features(
    text: str, offsets: list[tuple[int, int]]
) -> list[tuple[float, float, float, float, float]]:
    bounds, feats = clause_features(text)
    starts = [s for s, _ in bounds]
    out: list[tuple[float, float, float, float, float]] = []
    for tok_s, tok_e in offsets:
        if tok_e <= tok_s:
            out.append(ZERO5)
            continue
        i = bisect_right(starts, tok_s) - 1
        if i < 0:
            i = 0
        if i >= len(feats):
            i = len(feats) - 1
        out.append(feats[i])
    return out


def pack_window(
    input_ids: list[int],
    labels: list[int],
    offsets: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    cls_id: int,
    sep_id: int,
    pad_id: int,
    max_length: int,
    tok_features: list[tuple[float, float, float, float, float]] | None = None,
) -> dict[str, Any]:
    content_ids = input_ids[w_start:w_end]
    content_labs = labels[w_start:w_end]
    content_off = offsets[w_start:w_end]
    ids = [cls_id] + content_ids + [sep_id]
    labs = [IGNORE_LABEL] + content_labs + [IGNORE_LABEL]
    mask = [1] * len(ids)
    if len(ids) > max_length:
        raise SystemExit(f"window length {len(ids)} exceeds max_length {max_length}")
    pad_n = max_length - len(ids)
    if pad_n:
        ids = ids + [pad_id] * pad_n
        labs = labs + [IGNORE_LABEL] * pad_n
        mask = mask + [0] * pad_n
    real_offs = [(s, e) for s, e in content_off if e > s]
    char_start = real_offs[0][0] if real_offs else -1
    char_end = real_offs[-1][1] if real_offs else -1
    packed = {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": labs,
        "token_start": w_start,
        "token_end": w_end,
        "char_start": char_start,
        "char_end": char_end,
    }
    if tok_features is not None:
        content_f = tok_features[w_start:w_end]
        feats = [ZERO5] + list(content_f) + [ZERO5]
        if pad_n:
            feats = feats + [ZERO5] * pad_n
        packed["features"] = [list(x) for x in feats]
    return packed


def stratified_doc_split(
    rows: pd.DataFrame,
    seed: int,
) -> tuple[list[str], list[str], list[str], bool, str]:
    """80/10/10 at pecha level, stratified by coverage quartile when possible."""
    rng = random.Random(seed)
    work = rows[["pecha_id", "coverage_pct"]].copy()
    note = (
        "Document-level 80/10/10 split, stratified by Phase 2 coverage_pct "
        "quartiles so val/test are not accidentally all low- or high-density "
        f"(including outlier {HIGH_DENSITY_OUTLIER})."
    )
    try:
        work["stratum"] = pd.qcut(work["coverage_pct"], q=4, duplicates="drop")
        used_strata = True
    except ValueError:
        work["stratum"] = "all"
        used_strata = False
        note = (
            "Document-level 80/10/10 split (plain random within one bucket - "
            "qcut stratification was not possible)."
        )

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for _, group in work.groupby("stratum", observed=True, sort=False):
        ids = group["pecha_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * 0.10))
        n_val = int(round(n * 0.10))
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        while n_test + n_val >= n and n > 1:
            if n_test > n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break
        test.extend(ids[:n_test])
        val.extend(ids[n_test : n_test + n_val])
        train.extend(ids[n_test + n_val :])
    return train, val, test, used_strata, note


def token_stats(examples: list[dict[str, Any]]) -> tuple[int, int, float]:
    pos = 0
    total = 0
    for ex in examples:
        for lab, att in zip(ex["labels"], ex["attention_mask"]):
            if att != 1 or lab == IGNORE_LABEL:
                continue
            total += 1
            if lab != LABEL2ID["O"]:
                pos += 1
    pct = (100.0 * pos / total) if total else 0.0
    return pos, total, pct


def write_dataset_card(
    path: Path,
    *,
    source: str,
    n_new: int,
    n_old: int,
    split_ids: dict[str, list[str]],
    split_examples: dict[str, list[dict[str, Any]]],
    n_dropped: int,
    dropped_csv: Path,
    max_length: int,
    stride: int,
    seed: int,
    split_note: str,
    tokenizer_name: str,
    outlier_windows: dict[str, int],
    outlier_pos_share: dict[str, float],
    sidecar_note: str = "",
    quote_note: str = "",
    label2id: dict[str, int] | None = None,
) -> None:
    label2id = label2id or LABEL2ID
    lines = [
        "# tsawa binary BIO dataset",
        "",
        "Built by `src/build_tsawa_dataset.py` from Phase 2 `tsawa_audit.csv`.",
        "**No model was trained.** Data-rights clearance is still an open question.",
        "",
        "## Source",
        "",
        f"- `--source {source}`: **{n_new} new** + **{n_old} old** repos "
        f"(documents with a valid Tsawa layer per the audit).",
        f"- Tokenizer: `{tokenizer_name}`",
        f"- Windows: `max_length={max_length}`, `stride={stride}` "
        "(token-index based; special tokens wrap each window).",
        f"- Document split seed: `{seed}`",
        f"- {split_note}",
        "",
        "## Cleaning",
        "",
        sidecar_note or (
            f"- Dropped **{n_dropped}** zero-length spans (`start == end`).\n"
            f"- Log: `{dropped_csv}` (pecha_id, annotation_id, start, end)."
        ),
        "",
        "## Labels",
        "",
        f"- `{label2id}`",
        "- Special tokens and padding use label `-100`.",
        "- Token / span straddles: labeled by the token **start** offset.",
        "",
        "## `isverse`",
        "",
        "Not used. The new batch has a complete true/false `isverse` field; "
        "the old batch is entirely missing it (6,845 spans). Using it would "
        "encode batch identity, not verse structure.",
        "",
        f"## High-density outlier `{HIGH_DENSITY_OUTLIER}`",
        "",
        "Phase 2 coverage ≈ 96.7% tsawa. It is windowed with the same "
        "max_length/stride as every other document (no special case). "
        "It can still contribute a large share of positive tokens/windows:",
        "",
    ]
    for split_name in ("train", "validation", "test"):
        w = outlier_windows.get(split_name, 0)
        share = outlier_pos_share.get(split_name, 0.0)
        lines.append(
            f"- `{split_name}`: {w} windows from `{HIGH_DENSITY_OUTLIER}`; "
            f"{share:.1f}% of that split's positive (non-O) tokens."
        )
    lines += [
        "",
        "## Split composition",
        "",
        "| split | documents | windows | positive tokens | positive token % |",
        "|-------|----------:|--------:|----------------:|-----------------:|",
    ]
    for split_name in ("train", "validation", "test"):
        ex = split_examples[split_name]
        pos, total, pct = token_stats(ex)
        n_docs = len(split_ids[split_name])
        lines.append(
            f"| {split_name} | {n_docs} | {len(ex)} | {pos:,} / {total:,} | {pct:.3f}% |"
        )
    lines += [
        "",
        "Positive-token % is the class imbalance that matters for training; "
        "it is **not** the same as the document-level tsawa coverage % in the audit "
        "(~2.1% of all base text, ~4.5% among repos that have a Tsawa layer).",
        "",
        "### Documents per split",
        "",
    ]
    for split_name in ("train", "validation", "test"):
        ids = ", ".join(sorted(split_ids[split_name]))
        lines.append(f"- **{split_name}** ({len(split_ids[split_name])}): {ids}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_length < 8:
        raise SystemExit("--max-length is too small")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.stride >= args.max_length:
        print(
            "warning: stride >= max_length means windows will not overlap",
            file=sys.stderr,
        )

    audit_path = args.audit_csv.expanduser().resolve()
    audit = load_and_validate_audit(audit_path)
    selected = select_repos(audit, args.source)
    if args.limit is not None:
        selected = selected.head(args.limit)
        print(f"warning: --limit {args.limit} (debug); not a full build", file=sys.stderr)

    file_split_early = None
    if args.split_file is not None and args.restrict_to_split:
        file_split_early, _ = load_split_file(args.split_file.expanduser().resolve())
        selected = selected[selected["pecha_id"].astype(str).isin(file_split_early)].reset_index(drop=True)
        print(f"restrict-to-split: {len(selected)} documents")

    raw_dir = args.raw_dir.expanduser().resolve()
    expected = {"combined": 212, "new": 123, "old": 89}[args.source]
    if args.limit is None and not args.restrict_to_split and len(selected) != expected:
        raise SystemExit(
            f"--source {args.source} selected {len(selected)} repos, expected {expected}"
        )

    # Paths from the audit must still exist.
    missing_files: list[str] = []
    for rec in selected.itertuples(index=False):
        for attr in ("base_path", "tsawa_path"):
            p = Path(getattr(rec, attr))
            if not p.is_file():
                missing_files.append(f"{rec.pecha_id}: {attr} missing ({p})")
    if missing_files:
        raise SystemExit(
            "Audit paths no longer match data/raw_opf/:\n  "
            + "\n  ".join(missing_files[:20])
        )

    print(f"Audit OK ({PHASE2_TSAWA_REPOS} tsawa repos).")
    print(f"Building --source {args.source}: {len(selected)} documents")
    print(f"Tokenizer {args.tokenizer}  max_length={args.max_length} stride={args.stride}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit("A fast tokenizer is required for offset mapping.")
    # We window ourselves; do not let the tokenizer warn/truncate at 8192.
    tokenizer.model_max_length = int(1e12)
    cls_id, sep_id, pad_id = special_token_ids(tokenizer)
    n_special = 2  # CLS + SEP wrapping each window
    content_len = args.max_length - n_special
    if content_len < 1:
        raise SystemExit("max_length must leave room for CLS/SEP")

    if args.split_file is not None:
        split_path = args.split_file.expanduser().resolve()
        file_split, split_note = load_split_file(split_path)
        if args.restrict_to_split:
            pass  # already filtered
        selected_ids = list(selected["pecha_id"].astype(str))
        missing = [pid for pid in selected_ids if pid not in file_split]
        if missing:
            raise SystemExit(
                f"{split_path}: no split for {len(missing)} selected pecha(s): "
                + ", ".join(missing[:20])
            )
        extra = sorted(set(file_split) - set(selected_ids))
        if extra:
            print(
                f"note: split file lists {len(extra)} pecha(s) not in this "
                f"--source {args.source} selection (ignored)",
                file=sys.stderr,
            )
        split_of = {pid: file_split[pid] for pid in selected_ids}
        train_ids = [p for p in selected_ids if split_of[p] == "train"]
        val_ids = [p for p in selected_ids if split_of[p] == "validation"]
        test_ids = [p for p in selected_ids if split_of[p] == "test"]
        print(f"Split file: {split_path}")
    else:
        train_ids, val_ids, test_ids, _strat, split_note = stratified_doc_split(
            selected, args.seed
        )
        split_of = {pid: "train" for pid in train_ids}
        split_of.update({pid: "validation" for pid in val_ids})
        split_of.update({pid: "test" for pid in test_ids})
    if len(split_of) != len(selected):
        raise SystemExit("split assignment lost or duplicated a document")

    excluded = [p.strip() for p in args.exclude_pechas.split(",") if p.strip()]
    exclude_note = ""
    if excluded:
        dupes = sorted({p for p in excluded if excluded.count(p) > 1})
        if dupes:
            raise SystemExit(f"--exclude-pechas repeats: {', '.join(dupes)}")
        unknown = [p for p in excluded if p not in split_of]
        if unknown:
            raise SystemExit(
                "--exclude-pechas lists pecha(s) not in this build: " + ", ".join(unknown)
            )
        protected = [p for p in excluded if split_of[p] != "train"]
        if protected:
            raise SystemExit(
                "--exclude-pechas may only drop train documents; refusing to change "
                "the frozen eval splits: "
                + ", ".join(f"{p} ({split_of[p]})" for p in protected)
            )
        keep = set(split_of) - set(excluded)
        selected = selected[selected["pecha_id"].astype(str).isin(keep)].reset_index(drop=True)
        split_of = {p: s for p, s in split_of.items() if p in keep}
        train_ids = [p for p in train_ids if p in keep]
        exclude_note = (
            f"Excluded from train: {', '.join(sorted(excluded))} "
            f"({len(excluded)} document(s); validation and test unchanged)"
        )
        print(exclude_note)
        split_note = f"{split_note}. {exclude_note}"

    if not (train_ids and val_ids and test_ids):
        raise SystemExit("split assignment left a split empty")

    print(
        f"Document split: train={len(train_ids)} val={len(val_ids)} "
        f"test={len(test_ids)}"
    )

    use_sidecar = not args.from_yaml
    sidecar_by_repo: dict[str, list[tuple[int, int, str]]] = {}
    sidecar_dropped_all: list[dict[str, Any]] = []
    sidecar_note = ""
    if use_sidecar:
        sidecar_path = args.sidecar.expanduser().resolve()
        sidecar_by_repo, sidecar_dropped_all = load_sidecar_spans(sidecar_path)
        n_active = sum(len(v) for v in sidecar_by_repo.values())
        n_drop = len(sidecar_dropped_all)
        n_rows = n_active + n_drop
        print(
            f"Sidecar {sidecar_path.name}: {n_rows} rows, "
            f"{n_active} active, {n_drop} dropped"
        )
        if (
            args.limit is None
            and args.source == "combined"
            and sidecar_path.name == "tsawa_spans_resolved.csv"
        ):
            if n_rows != PHASE3_SIDECAR_ROWS or n_drop != PHASE3_SIDECAR_DROPPED:
                raise SystemExit(
                    f"sidecar totals drifted: rows={n_rows} dropped={n_drop}, "
                    f"expected {PHASE3_SIDECAR_ROWS}/{PHASE3_SIDECAR_DROPPED}"
                )
            if n_active != PHASE3_SIDECAR_ACTIVE:
                raise SystemExit(
                    f"sidecar active spans={n_active}, expected {PHASE3_SIDECAR_ACTIVE}"
                )
        selected_ids = set(selected["pecha_id"].astype(str))
        sidecar_ids = set(sidecar_by_repo) | {r["pecha_id"] for r in sidecar_dropped_all}
        missing_ids = sorted(selected_ids - sidecar_ids)
        if missing_ids:
            raise SystemExit(
                "Sidecar missing pecha_id(s): " + ", ".join(missing_ids[:20])
            )
        extra = ""
        if sidecar_path.name == "tsawa_spans_merged.csv":
            extra = (
                "- Adjacent tsawa spans whose gap is only shad / tsheg / whitespace "
                "were merged (`src/merge_tsawa_spans.py`).\n"
            )
        sidecar_note = (
            f"- Labels from `{sidecar_path}`.\n"
            f"{extra}"
            f"- Skip `dropped=True` (and empty stubs). **{n_drop}** sidecar rows excluded; "
            f"**{n_active}** spans labeled.\n"
            f"- Cloned `Tsawa.yml` is not rewritten.\n"
            f"- Log of excluded sidecar rows: `{args.dropped_csv}`."
        )

    multiclass = args.quote_sidecar is not None
    label2id = LABEL2ID_5 if multiclass else LABEL2ID
    id2label = ID2LABEL_5 if multiclass else ID2LABEL
    quote_by_repo: dict[str, list[tuple[int, int, str]]] = {}
    precedence_stats: Counter = Counter()
    docs_with_quote = 0
    quote_note = ""
    if multiclass:
        qpath = args.quote_sidecar.expanduser().resolve()
        if not qpath.is_file():
            raise SystemExit(f"Quote sidecar not found: {qpath}")
        n_q_rows = n_q_drop = 0
        with qpath.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                n_q_rows += 1
                if str(r.get("dropped", "")).lower() == "true":
                    n_q_drop += 1
                    continue
                s, e = int(r["start"]), int(r["end"])
                if e <= s:
                    n_q_drop += 1
                    continue
                quote_by_repo.setdefault(r["pecha_id"], []).append((s, e, r["ann_id"]))
        for v in quote_by_repo.values():
            v.sort()
        n_q_active = n_q_rows - n_q_drop
        print(
            f"Quote sidecar {qpath.name}: {n_q_rows:,} rows, {n_q_active:,} active, "
            f"{n_q_drop} dropped, {len(quote_by_repo)} books"
        )
        print("5-label build: O / B-TSAWA / I-TSAWA / B-QUOTE / I-QUOTE (TSAWA wins overlaps)")
        quote_note = (
            f"- Quotation labels from `{qpath}` (merged `Quotation.yml` old batch + "
            f"`Citation.yml` new batch, snapped by `src/snap_quotation_boundaries.py`).\n"
            f"- **{n_q_active:,}** quotation spans over **{len(quote_by_repo)}** books; "
            f"{n_q_drop} sidecar rows dropped as quote-quote overlap stubs.\n"
            "- **Precedence: TSAWA wins.** Where a quotation span overlaps a tsawa span "
            "the overlapping characters are labeled TSAWA and the quotation span is "
            "carved around them."
        )

    ignore_by_repo: dict[str, list[tuple[int, int]]] = {}
    if args.ignore_spans_csv is not None:
        ip = args.ignore_spans_csv.expanduser().resolve()
        if not ip.is_file():
            raise SystemExit(f"ignore-spans csv not found: {ip}")
        with ip.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ignore_by_repo.setdefault(r["pecha_id"], []).append(
                    (int(r["start"]), int(r["end"]))
                )
        for v in ignore_by_repo.values():
            v.sort()
        print(f"Ignore spans: {sum(len(v) for v in ignore_by_repo.values()):,} "
              f"across {len(ignore_by_repo)} books")

    dropped_rows: list[dict[str, Any]] = []
    split_examples: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    selected_id_set = set(selected["pecha_id"].astype(str))
    if use_sidecar:
        dropped_rows.extend(
            r for r in sidecar_dropped_all if r["pecha_id"] in selected_id_set
        )

    for i, rec in enumerate(selected.itertuples(index=False), start=1):
        pecha_id = rec.pecha_id
        base_path = Path(rec.base_path)
        tsawa_path = Path(rec.tsawa_path)
        if not base_path.is_file() or not tsawa_path.is_file():
            # Should be unreachable after the pre-check; keep it loud.
            raise SystemExit(f"{pecha_id}: missing base/tsawa after pre-check")
        text = base_path.read_text(encoding="utf-8")
        if use_sidecar:
            spans = sidecar_by_repo.get(pecha_id, [])
        else:
            spans, dropped = load_cleaned_spans(tsawa_path, pecha_id)
            dropped_rows.extend(dropped)
            expected_drop = int(rec.n_zero_length)
            if len(dropped) != expected_drop:
                raise SystemExit(
                    f"{pecha_id}: dropped {len(dropped)} zero-length spans, "
                    f"audit n_zero_length={expected_drop}"
                )

        tsawa_only = spans
        if multiclass:
            quote_spans = quote_by_repo.get(pecha_id, [])
            spans, prec = apply_tsawa_precedence(spans, quote_spans)
            for k, v in prec.items():
                precedence_stats[k] += v
            if quote_spans:
                docs_with_quote += 1

        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        input_ids = list(enc["input_ids"])
        offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        labels = label_tokens(offsets, spans, label2id)
        ign = merge_intervals(ignore_by_repo.get(pecha_id, []))
        if ign:
            for ti, (tok_s, tok_e) in enumerate(offsets):
                if tok_e > tok_s and in_intervals(ign, tok_s) and labels[ti] != LABEL2ID["O"]:
                    labels[ti] = IGNORE_LABEL
        tok_feats = token_clause_features(text, offsets) if args.add_features else None

        if multiclass and quote_spans:
            # Tokens the raw quotation layer claimed but TSAWA took.
            tsa = merge_intervals((s, e) for s, e, _a in tsawa_only)
            qsm = merge_intervals((s, e) for s, e, _a in quote_spans)
            for tok_s, tok_e in offsets:
                if tok_e > tok_s and in_intervals(qsm, tok_s) and in_intervals(tsa, tok_s):
                    precedence_stats["tokens_suppressed"] += 1
        windows = sliding_windows(len(input_ids), content_len, args.stride)
        split_name = split_of[pecha_id]
        for w_i, (w_s, w_e) in enumerate(windows):
            packed = pack_window(
                input_ids,
                labels,
                offsets,
                w_s,
                w_e,
                cls_id,
                sep_id,
                pad_id,
                args.max_length,
                tok_features=tok_feats,
            )
            packed.update(
                {
                    "pecha_id": pecha_id,
                    "source_batch": rec.source_batch,
                    "window_index": w_i,
                    "n_tokens_doc": len(input_ids),
                    "coverage_pct": float(rec.coverage_pct),
                }
            )
            split_examples[split_name].append(packed)
        if i == 1 or i == len(selected) or i % 20 == 0:
            print(
                f"  [{i}/{len(selected)}] {pecha_id}  "
                f"tokens={len(input_ids)} windows={len(windows)} → {split_name}",
                file=sys.stderr,
            )

    dropped_csv = args.dropped_csv.expanduser().resolve()
    dropped_csv.parent.mkdir(parents=True, exist_ok=True)
    with dropped_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["pecha_id", "annotation_id", "start", "end"]
        )
        writer.writeheader()
        writer.writerows(dropped_rows)
    if (
        not use_sidecar
        and args.limit is None
        and args.source == "combined"
        and len(dropped_rows) != PHASE2_ZERO_LENGTH_SPANS
    ):
        raise SystemExit(
            f"dropped_spans.csv has {len(dropped_rows)} rows, "
            f"expected {PHASE2_ZERO_LENGTH_SPANS}"
        )

    feat_schema = {
            "pecha_id": Value("string"),
            "source_batch": Value("string"),
            "window_index": Value("int32"),
            "n_tokens_doc": Value("int32"),
            "coverage_pct": Value("float64"),
            "token_start": Value("int32"),
            "token_end": Value("int32"),
            "char_start": Value("int32"),
            "char_end": Value("int32"),
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "labels": Sequence(Value("int32")),
        }
    if args.add_features:
        feat_schema["features"] = Sequence(Sequence(Value("float32")))
    features = Features(feat_schema)
    def _to_dataset(examples: list[dict[str, Any]]) -> Dataset:
        if examples:
            return Dataset.from_list(examples, features=features)
        empty = {name: [] for name in features}
        return Dataset.from_dict(empty, features=features)

    dset = DatasetDict(
        {name: _to_dataset(examples) for name, examples in split_examples.items()}
    )
    # Persist label maps with the dataset for later training.
    dset.info = dset["train"].info  # keep default; maps go on the card + a sidecar
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        # save_to_disk refuses a non-empty dir in some versions; overwrite cleanly.
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dset.save_to_disk(str(out_dir))
    (out_dir / "label_map.json").write_text(
        __import__("json").dumps(
            {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    def outlier_window_count(split: str) -> int:
        return sum(1 for ex in split_examples[split] if ex["pecha_id"] == HIGH_DENSITY_OUTLIER)

    def outlier_pos_share(split: str) -> float:
        pos_out = 0
        pos_all = 0
        for ex in split_examples[split]:
            for lab, att in zip(ex["labels"], ex["attention_mask"]):
                if att != 1 or lab == IGNORE_LABEL or lab == LABEL2ID["O"]:
                    continue
                pos_all += 1
                if ex["pecha_id"] == HIGH_DENSITY_OUTLIER:
                    pos_out += 1
        return (100.0 * pos_out / pos_all) if pos_all else 0.0

    n_new = int((selected["source_batch"] == "new").sum())
    n_old = int((selected["source_batch"] == "old").sum())
    write_dataset_card(
        out_dir / "dataset_card.md",
        source=args.source,
        n_new=n_new,
        n_old=n_old,
        split_ids={"train": train_ids, "validation": val_ids, "test": test_ids},
        split_examples=split_examples,
        n_dropped=len(dropped_rows),
        dropped_csv=dropped_csv,
        max_length=args.max_length,
        stride=args.stride,
        seed=args.seed,
        split_note=split_note,
        tokenizer_name=args.tokenizer,
        outlier_windows={s: outlier_window_count(s) for s in split_examples},
        outlier_pos_share={s: outlier_pos_share(s) for s in split_examples},
        sidecar_note=sidecar_note + ("\n" + quote_note if quote_note else ""),
        label2id=label2id,
    )

    print("\n=== tsawa dataset ===")
    print(f"Repos included : {len(selected)}  (new={n_new}, old={n_old})")
    print(f"Dropped spans  : {len(dropped_rows)} → {dropped_csv}")
    print(f"Saved          : {out_dir}")
    for name in ("train", "validation", "test"):
        pos, total, pct = token_stats(split_examples[name])
        print(
            f"  {name:12s} docs={len({'train': train_ids, 'validation': val_ids, 'test': test_ids}[name]):3d}  "
            f"windows={len(split_examples[name]):5d}  "
            f"pos-tokens={pos:,}/{total:,} ({pct:.3f}%)"
        )
    if multiclass:
        s = precedence_stats
        print(f"\n=== TSAWA-wins precedence ({docs_with_quote} books with a quote layer) ===")
        print(f"  quotation spans in        : {s['quote_spans_in']:,}")
        print(f"  untouched by a tsawa span : "
              f"{s['quote_spans_in'] - s['quote_spans_touched']:,}")
        print(f"  clipped by a tsawa span   : {s['quote_spans_clipped']:,}"
              f"  (of which split in two: {s['quote_spans_fragmented']:,})")
        print(f"  fully swallowed (dropped) : {s['quote_spans_dropped']:,}")
        print(f"  quotation spans out       : {s['quote_spans_out']:,}")
        print(f"  characters suppressed     : {s['chars_suppressed']:,}")
        print(f"  TOKENS suppressed         : {s['tokens_suppressed']:,}")
    print("No training was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
