#!/usr/bin/env python3
"""
Frozen document split for the Sabche dataset (83/8.5/8.5 by window count).
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from build_tsawa_dataset import sliding_windows  # noqa: E402

AUDIT = ROOT / "data/tsawa_audit.csv"
CLEAN = ROOT / "data/sabche_spans_clean.csv"
VERDICTS = ROOT / "data/sabche_book_verdicts.csv"
SPLIT_V3 = ROOT / "data/tsawa_split_frozen.csv"
OUT_EXCL = ROOT / "data/sabche_excluded_books.csv"
TOKENIZER = "jhu-clsp/mmBERT-base"
CONTENT_LEN = 8190  # 8192 - CLS - SEP
STRIDE = 5120

UNDER_ANNOTATED = {  # titled-heading recall < 0.2 (outline/TOC only)
    "I7A79CC95": 0.00, "IB8F4E5BE": 0.00, "P000031": 0.01,
    "I100E7DAD": 0.14, "I9779A606": 0.19,
}
MATCH_MIN = 40
LINK_SHARE = 0.10
MIN_SHARED = 2
MIN_SHARED_ABS = 10  # large compilations share many headings at < LINK_SHARE
K = 6
SEED = 123
PRIORITY = {"test": 0, "val": 1, "train": 2}

WS = re.compile(r"\s+")
SYL = re.compile(r"[་-༔\s]+")


def syls(s: str) -> list[str]:
    return [x for x in SYL.split(s) if x]


class UF:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Frozen Sabche document split.")
    p.add_argument("--val-frac", type=float, default=0.085)
    p.add_argument("--test-frac", type=float, default=0.085)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    TARGETS = {"train": 1 - args.val_frac - args.test_frac,
               "val": args.val_frac, "test": args.test_frac}
    OUT_SPLIT = ROOT / "data/split_frozen.csv"
    OUT_REPORT = ROOT / "results/sabche_split_report.json"
    audit = {r["pecha_id"]: r for r in csv.DictReader(AUDIT.open(encoding="utf-8"))}
    verdict = {r["pecha_id"]: r for r in csv.DictReader(VERDICTS.open(encoding="utf-8"))}
    v3 = {r["pecha_id"]: r for r in csv.DictReader(
        l for l in SPLIT_V3.open(encoding="utf-8") if not l.startswith("#"))}

    excl = []
    for pid, v in sorted(verdict.items()):
        if v["verdict"] == "exclude":
            excl.append({"pecha_id": pid, "batch": v["batch"], "n_spans": v["n_spans"],
                         "reason": "offsets_unrecoverable" if v["batch"] == "old"
                         else "no_heading_like_spans",
                         "evidence": f"heading_rate={v['heading_rate_final']}"})
        elif pid in UNDER_ANNOTATED:
            excl.append({"pecha_id": pid, "batch": v["batch"], "n_spans": v["n_spans"],
                         "reason": "under_annotated_body_headings",
                         "evidence": f"titled_heading_recall={UNDER_ANNOTATED[pid]}"})
    excluded = {e["pecha_id"] for e in excl}

    spans = defaultdict(list)
    for r in csv.DictReader(CLEAN.open(encoding="utf-8")):
        if r["dropped"] == "True" or r["pecha_id"] in excluded:
            continue
        spans[r["pecha_id"]].append((int(r["start"]), int(r["end"])))
    books = sorted(spans)
    print(f"books kept {len(books)}  excluded {len(excluded)}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    tok.model_max_length = int(1e12)
    texts, stripped, n_tok, n_win = {}, {}, {}, {}
    for i, p in enumerate(books):
        t = Path(audit[p]["base_path"]).read_text(encoding="utf-8")
        texts[p], stripped[p] = t, WS.sub("", t)
        n_tok[p] = len(tok(t, add_special_tokens=False)["input_ids"])
        n_win[p] = len(sliding_windows(n_tok[p], CONTENT_LEN, STRIDE))
        if (i + 1) % 50 == 0:
            print(f"  tokenized {i + 1}/{len(books)}", flush=True)

    # verbatim sharing of long Sabche spans
    needles = {p: [] for p in books}
    for p in books:
        for s, e in spans[p]:
            if e - s >= MATCH_MIN:
                nd = WS.sub("", texts[p][s:e])
                if len(syls(nd)) >= K + 2:
                    needles[p].append((nd, e - s))
    key_index = defaultdict(list)
    for p, ns in needles.items():
        for i, (nd, _) in enumerate(ns):
            key_index[hash(tuple(syls(nd)[1:1 + K]))].append((p, i))
    found = defaultdict(set)
    for p2 in books:
        sy = syls(stripped[p2])
        cand = set()
        for j in range(len(sy) - K + 1):
            h = hash(tuple(sy[j:j + K]))
            if h in key_index:
                cand.update(key_index[h])
        for p, i in cand:
            if p != p2 and needles[p][i][0] in stripped[p2]:
                found[(p, i)].add(p2)
    share, nshared = Counter(), Counter()
    for (p, i), others in found.items():
        for p2 in others:
            share[(p, p2)] += needles[p][i][1]
            nshared[(p, p2)] += 1
    tot = {p: sum(L for _, L in v) or 1 for p, v in needles.items()}
    links = sorted({tuple(sorted((a, b))) for (a, b), c in share.items()
                    if (c / tot[a] >= LINK_SHARE and nshared[(a, b)] >= MIN_SHARED)
                    or nshared[(a, b)] >= MIN_SHARED_ABS})
    boiler = sorted({tuple(sorted((a, b))) for (a, b), c in share.items()
                     if c / tot[a] >= LINK_SHARE and nshared[(a, b)] < MIN_SHARED}
                    - set(links))
    print(f"sabche links {len(links)} (single-span links ignored: {len(boiler)})")

    uf = UF(books)
    by_v3g = defaultdict(list)
    for p in books:
        if p in v3:
            by_v3g[v3[p]["group_id"]].append(p)
    for mem in by_v3g.values():
        for q in mem[1:]:
            uf.union(mem[0], q)
    for a, b in links:
        uf.union(a, b)
    groups = defaultdict(list)
    for p in books:
        groups[uf.find(p)].append(p)

    assign, moved, conflicts = {}, [], []
    free = []
    for root, mem in groups.items():
        fixed = {v3[p]["split"] for p in mem if p in v3}
        if not fixed:
            free.append(mem)
            continue
        sp = min(fixed, key=PRIORITY.get)
        if len(fixed) > 1:
            conflicts.append({"members": sorted(mem), "fixed_splits": sorted(fixed), "to": sp})
        for p in mem:
            assign[p] = sp
            if p in v3 and v3[p]["split"] != sp:
                moved.append((p, v3[p]["split"], sp))

    rng = random.Random(SEED)
    free.sort(key=lambda m: (-sum(n_win[p] for p in m), rng.random()))
    total_w = sum(n_win.values())
    cur = Counter()
    for p, sp in assign.items():
        cur[sp] += n_win[p]
    for mem in free:
        w = sum(n_win[p] for p in mem)
        sp = max(TARGETS, key=lambda s: (TARGETS[s] * total_w - cur[s]) / TARGETS[s])
        for p in mem:
            assign[p] = sp
        cur[sp] += w

    # leakage on the final split
    train = {p for p in books if assign[p] == "train"}
    leak = {}
    for spn in ("val", "test"):
        nl = nh = 0
        for p in books:
            if assign[p] != spn:
                continue
            for i in range(len(needles[p])):
                nl += 1
                nh += bool(found.get((p, i), set()) & train)
        leak[spn] = {"n_long": nl, "n_in_train": nh, "pct": round(100 * nh / nl, 2) if nl else 0}
    straddle = [(a, b) for a, b in links if assign[a] != assign[b]]

    stats = {}
    for sp in ("train", "val", "test"):
        m = [p for p in books if assign[p] == sp]
        stats[sp] = {
            "books": len(m), "old": sum(p.startswith("P") for p in m),
            "new": sum(p.startswith("I") for p in m),
            "windows": sum(n_win[p] for p in m),
            "pct_windows": round(100 * sum(n_win[p] for p in m) / total_w, 1),
            "sabche_spans": sum(len(spans[p]) for p in m),
            "from_split_v3": sum(p in v3 for p in m),
        }
    for sp, s in stats.items():
        print(f"{sp:5} books={s['books']:3} (old {s['old']}, new {s['new']}, v3 {s['from_split_v3']}) "
              f"windows={s['windows']} ({s['pct_windows']}%) spans={s['sabche_spans']:,}")
    print(f"conflicting groups: {len(conflicts)}; v3 books moved: {moved}")
    for spn, l in leak.items():
        print(f"leakage {spn}: {l['n_in_train']}/{l['n_long']} = {l['pct']}%")
    print(f"linked pairs straddling splits: {len(straddle)}")

    gid = {p: min(groups[uf.find(p)]) for p in books}
    with OUT_SPLIT.open("w", newline="", encoding="utf-8") as fh:
        fh.write(
            f"# sabche layer-detection document split {args.version} - FROZEN\n"
            f"# generated: {date.today().isoformat()}\n"
            "# generator: src/prepare_sabche_split.py\n"
            f"# seed: {SEED}\n"
            "# base: tsawa_split_frozen.csv assignments kept; conflicting groups -> test>val>train\n"
            f"# must-link: split_v3 group_id UNION sabche verbatim share >= {LINK_SHARE} "
            f"over >= {MIN_SHARED} spans, or >= {MIN_SHARED_ABS} shared spans (>= {MATCH_MIN} chars)\n"
            f"# targets for new groups: {TARGETS['train']:.0%}/{TARGETS['val']:.1%}/{TARGETS['test']:.1%} "
            "by WINDOW count (8192 / 5120)\n"
            "# span source: data/sabche_spans_clean.csv (dropped=False)\n"
            "# TEST SPLIT IS FROZEN: evaluate test once at the end, never for tuning.\n")
        w = csv.DictWriter(fh, fieldnames=["pecha_id", "split", "group_id", "n_tokens", "n_windows"])
        w.writeheader()
        for p in books:
            w.writerow({"pecha_id": p, "split": assign[p], "group_id": gid[p],
                        "n_tokens": n_tok[p], "n_windows": n_win[p]})
    with OUT_EXCL.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(excl[0]))
        w.writeheader()
        w.writerows(excl)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps({
        "stats": stats, "leak": leak, "n_links": len(links), "n_boilerplate_links": len(boiler),
        "links": links, "boilerplate_links": boiler, "conflicts": conflicts, "moved": moved,
        "straddle": straddle, "excluded": excl,
        "n_groups": len(groups), "n_multi_groups": sum(len(m) > 1 for m in groups.values()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_SPLIT}\nwrote {OUT_EXCL}\nwrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
