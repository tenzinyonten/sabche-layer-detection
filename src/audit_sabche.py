#!/usr/bin/env python3
"""
Pre-build audit of the Sabche layer (deliverable A). Read-only.
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
from check_boundary_snapping import check_span  # noqa: E402

AUDIT = ROOT / "data/tsawa_audit.csv"
SPLIT_V3 = ROOT / "data/tsawa_split_frozen.csv"
OUT = ROOT / "results/audit/sabche_audit.csv"
OUT_BOOKS = ROOT / "results/audit/sabche_books.csv"
OTHER = ("Chapter.yml", "Tsawa.yml", "Quotation.yml", "Citation.yml",
         "Commentary.yml", "Yigchung.yml", "BookTitle.yml", "Author.yml")
MATCH_MIN = 40
LINK_SHARE = 0.10
K = 6  # syllable shingle size for leakage candidate lookup

PUNCT_GAP = re.compile(r"^[\s་-༔༺-༽\xa0]*$")
WS = re.compile(r"\s+")
SYL = re.compile(r"[་-༔\s]+")
ORD = ("དང་པོ", "གཉིས་པ", "གསུམ་པ", "བཞི་པ", "ལྔ་པ", "དྲུག་པ", "བདུན་པ",
       "བརྒྱད་པ", "དགུ་པ", "བཅུ་པ")
CLOSERS = ("ནི", "ལ", "ལའང", "ལ་ཡང", "ནི་འདི", "ཏེ", "བ", "པ", "དོ", "ཞེས")


def load_layer(opf: Path, name: str) -> list[tuple[int, int]]:
    p = opf / "layers/v001" / name
    if not p.is_file():
        return []
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = []
    anns = d.get("annotations") or {}
    if isinstance(anns, dict):
        anns = anns.values()
    for a in anns:
        if not isinstance(a, dict):
            continue
        sp = a.get("span") or {}
        s, e = sp.get("start"), sp.get("end")
        if s is None or e is None or int(e) <= int(s):
            continue
        out.append((int(s), int(e)))
    return sorted(out)


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def q(v, p):
    return statistics.quantiles(v, n=100)[p - 1] if len(v) > 100 else float("nan")


def syls(s: str) -> list[str]:
    return [x for x in SYL.split(s) if x]


def main() -> int:
    audit = {r["pecha_id"]: r for r in csv.DictReader(AUDIT.open(encoding="utf-8"))}
    split = {}
    for r in csv.DictReader(l for l in SPLIT_V3.open(encoding="utf-8") if not l.startswith("#")):
        split[r["pecha_id"]] = r

    books = {}
    for pid, r in audit.items():
        opf = Path(r["opf_root"])
        sab = load_layer(opf, "Sabche.yml")
        books[pid] = {
            "batch": "old" if pid.startswith("P") else "new",
            "opf": opf, "base": Path(r["base_path"]), "sab": sab,
            "has_file": (opf / "layers/v001/Sabche.yml").is_file(),
        }

    B = ("old", "new")
    n_books = Counter(b["batch"] for b in books.values())
    n_file = Counter(b["batch"] for b in books.values() if b["has_file"])
    n_nonempty = Counter(b["batch"] for b in books.values() if b["sab"])
    print("=== COVERAGE ===")
    print(f"{'batch':6} {'books':>6} {'Sabche.yml':>11} {'non-empty':>10} {'spans':>8}")
    for bt in B:
        ns = sum(len(b["sab"]) for b in books.values() if b["batch"] == bt)
        print(f"{bt:6} {n_books[bt]:6} {n_file[bt]:11} {n_nonempty[bt]:10} {ns:8,}")

    lens = defaultdict(list)
    frag = Counter(); adj = Counter(); gap0 = Counter()
    ov_spans = defaultdict(Counter); ov_any = Counter()
    self_ov = Counter(); nested = Counter()
    first_syl = defaultdict(Counter); last_syl = defaultdict(Counter)
    ord_open = Counter(); ni_close = Counter(); la_close = Counter(); shad_end = Counter()
    dirty = defaultdict(Counter)
    rows, book_rows = [], []
    texts = {}
    needles = {}  # pid -> list[(stripped, raw_len)]

    for pid, b in sorted(books.items()):
        if not b["sab"]:
            continue
        bt = b["batch"]
        text = b["base"].read_text(encoding="utf-8")
        texts[pid] = text
        sab = b["sab"]
        others = {n: load_layer(b["opf"], n) for n in OTHER}
        ns = []
        for i, (s, e) in enumerate(sab):
            ln = e - s
            lens[bt].append(ln)
            seg = text[s:e]
            # fragmentation: next Sabche span separated only by punct/ws
            is_frag = False
            if i + 1 < len(sab):
                s2, e2 = sab[i + 1]
                if s2 >= e:
                    gap = text[e:s2]
                    if PUNCT_GAP.match(gap):
                        is_frag = True
                        frag[bt] += 1
                        if s2 == e:
                            gap0[bt] += 1
                if s2 < e:
                    self_ov[bt] += 1
                    if e2 <= e:
                        nested[bt] += 1
            adj[bt] += 1
            hit = []
            for n, lst in others.items():
                if any(os_ < e and s < oe for os_, oe in lst):
                    ov_spans[bt][n] += 1
                    hit.append(n)
            if any(n in hit for n in ("Chapter.yml", "Tsawa.yml", "Quotation.yml", "Citation.yml")):
                ov_any[bt] += 1
            sy = syls(seg)
            if sy:
                first_syl[bt][sy[0]] += 1
                last_syl[bt][sy[-1]] += 1
            core = seg.strip().rstrip("།༎ ་")
            if any(seg.lstrip("། ").startswith(o) for o in ORD):
                ord_open[bt] += 1
            if core.endswith("ནི"):
                ni_close[bt] += 1
            if core.endswith("ལ") or core.endswith("ལའང"):
                la_close[bt] += 1
            if seg.rstrip().endswith("།"):
                shad_end[bt] += 1
            fl = check_span(text, s, e)
            ds, de = not fl["start_is_clean"], not fl["end_is_clean"]
            dirty[bt]["total"] += 1
            dirty[bt]["start"] += ds
            dirty[bt]["end"] += de
            dirty[bt]["either"] += ds or de
            rows.append({"pecha_id": pid, "batch": bt, "start": s, "end": e, "length": ln,
                         "dirty_start": ds, "dirty_end": de, "frag_next": is_frag,
                         "overlaps": "|".join(h.replace(".yml", "") for h in hit),
                         "text": seg[:80].replace("\n", " ")})
            if ln >= MATCH_MIN:
                nd = WS.sub("", seg)
                if len(syls(nd)) >= K + 2:
                    ns.append((nd, ln))
        needles[pid] = ns
        nch = len(text)
        book_rows.append({
            "pecha_id": pid, "batch": bt, "n_sabche": len(sab), "base_chars": nch,
            "per_100k": round(1e5 * len(sab) / max(nch, 1), 2),
            "sab_chars": sum(e - s for s, e in sab),
            "has_tsawa": bool(others["Tsawa.yml"]), "has_commentary": bool(others["Commentary.yml"]),
            "n_chapter": len(others["Chapter.yml"]),
            "in_split_v3": split.get(pid, {}).get("split", ""),
        })

    print("\n=== LENGTH (chars) ===")
    print(f"{'batch':6} {'n':>7} {'p10':>5} {'p25':>5} {'med':>5} {'p75':>5} {'p90':>5} {'p99':>6} {'max':>7} {'<15':>6} {'>300':>6}")
    for bt in B:
        v = lens[bt]
        print(f"{bt:6} {len(v):7,} {q(v,10):5.0f} {q(v,25):5.0f} {statistics.median(v):5.0f} "
              f"{q(v,75):5.0f} {q(v,90):5.0f} {q(v,99):6.0f} {max(v):7,} "
              f"{pct(sum(x<15 for x in v), len(v)):>6} {pct(sum(x>300 for x in v), len(v)):>6}")

    print("\n=== FRAGMENTATION / SELF-OVERLAP ===")
    print(f"{'batch':6} {'next sep by punct only':>24} {'gap=0':>8} {'overlap next':>13} {'nested':>8}")
    for bt in B:
        t = adj[bt]
        print(f"{bt:6} {frag[bt]:>10,} ({pct(frag[bt], t):>6}) {gap0[bt]:8,} "
              f"{self_ov[bt]:13,} {nested[bt]:8,}")

    print("\n=== CROSS-LAYER OVERLAP (% of Sabche spans touching layer) ===")
    print(f"{'layer':16} {'old':>8} {'new':>8}")
    for n in OTHER:
        print(f"{n.replace('.yml',''):16} " + " ".join(f"{pct(ov_spans[bt][n], adj[bt]):>8}" for bt in B))
    print(f"{'Chap|Tsa|Quo|Cit':16} " + " ".join(f"{pct(ov_any[bt], adj[bt]):>8}" for bt in B))

    print("\n=== LEXICAL MARKERS ===")
    print(f"{'marker':28} {'old':>8} {'new':>8}")
    for lab, c in (("opens w/ ordinal (དང་པོ…)", ord_open), ("closes ...ནི", ni_close),
                   ("closes ...ལ / ལའང", la_close), ("ends with shad ། (incl)", shad_end)):
        print(f"{lab:28} " + " ".join(f"{pct(c[bt], adj[bt]):>8}" for bt in B))
    for bt in B:
        print(f"{bt} top last syl: " + ", ".join(f"{k}:{pct(v, adj[bt])}" for k, v in last_syl[bt].most_common(8)))
        print(f"{bt} top first syl: " + ", ".join(f"{k}:{pct(v, adj[bt])}" for k, v in first_syl[bt].most_common(8)))

    print("\n=== BOUNDARY CLEANLINESS (raw offsets) ===")
    print(f"{'batch':6} {'total':>7} {'dirty start':>14} {'dirty end':>14} {'dirty either':>14}")
    for bt in list(B) + ["all"]:
        d = dirty[bt] if bt != "all" else dirty["old"] + dirty["new"]
        print(f"{bt:6} {d['total']:7,} " + " ".join(f"{d[k]:>6,} ({pct(d[k], d['total']):>5})" for k in ("start", "end", "either")))

    print("\n=== PER-BOOK DENSITY (non-empty Sabche books) ===")
    for bt in B:
        br = [r for r in book_rows if r["batch"] == bt]
        cnt = [r["n_sabche"] for r in br]
        dens = [r["per_100k"] for r in br]
        print(f"{bt:6} books={len(br)} spans/book med={statistics.median(cnt):.0f} "
              f"p10={q(cnt,10):.0f}  per100k med={statistics.median(dens):.1f} p10={q(dens,10):.1f}  "
              f"<10 spans: {sum(c<10 for c in cnt)}  <5 per100k: {sum(d<5 for d in dens)}")
    noyml_comm = [p for p, b in books.items() if not b["sab"] and (
        load_layer(b["opf"], "Tsawa.yml") or load_layer(b["opf"], "Commentary.yml"))]
    print(f"books w/ Tsawa or Commentary but NO Sabche: {len(noyml_comm)} "
          f"(old {sum(p.startswith('P') for p in noyml_comm)}, new {sum(p.startswith('I') for p in noyml_comm)})")

    # leakage
    print("\n=== LEAKAGE (Sabche spans >=40 chars, verbatim in another book) ===")
    stripped = {p: WS.sub("", t) for p, t in texts.items()}
    all_stripped = {p: WS.sub("", b["base"].read_text(encoding="utf-8"))
                    for p, b in books.items()}
    key_index = defaultdict(list)  # shingle hash -> [(pid, idx)]
    for p, ns in needles.items():
        for i, (nd, _) in enumerate(ns):
            key_index[hash(tuple(syls(nd)[1:1 + K]))].append((p, i))
    found = defaultdict(set)  # (pid, idx) -> other pids containing it
    for p2, st in all_stripped.items():
        sy = syls(st)
        cand = set()
        for j in range(len(sy) - K + 1):
            h = hash(tuple(sy[j:j + K]))
            if h in key_index:
                cand.update(key_index[h])
        for (p, i) in cand:
            if p == p2:
                continue
            if needles[p][i][0] in st:
                found[(p, i)].add(p2)
    n_long = sum(len(v) for v in needles.values())
    n_rep = len(found)
    print(f"long Sabche spans: {n_long:,}; found verbatim in >=1 other book: {n_rep:,} ({pct(n_rep, n_long)})")
    for bt in B:
        nl = sum(len(v) for p, v in needles.items() if books[p]["batch"] == bt)
        nr = sum(1 for (p, _) in found if books[p]["batch"] == bt)
        print(f"  {bt}: {nr:,}/{nl:,} ({pct(nr, nl)})")
    # book-pair share (same rule as prepare_v6: >=10% of long span chars, either direction)
    share = defaultdict(int)
    for (p, i), others in found.items():
        for p2 in others:
            share[(p, p2)] += needles[p][i][1]
    tot = {p: sum(L for _, L in v) or 1 for p, v in needles.items()}
    links = {(a, b) for (a, b), c in share.items() if c / tot[a] >= LINK_SHARE}
    pairs = {tuple(sorted(x)) for x in links}
    print(f"book pairs linked (>=10% share): {len(pairs)}")
    # against split_v3
    sabche_in_split = [p for p in needles if p in split]
    print(f"Sabche books in split_v3: {len(sabche_in_split)}/{len(needles)} "
          f"(train {sum(split[p]['split']=='train' for p in sabche_in_split)}, "
          f"val {sum(split[p]['split']=='val' for p in sabche_in_split)}, "
          f"test {sum(split[p]['split']=='test' for p in sabche_in_split)})")
    train = {p for p, r in split.items() if r["split"] == "train"}
    for spn in ("val", "test"):
        nl = nh = 0
        for p in sabche_in_split:
            if split[p]["split"] != spn:
                continue
            for i in range(len(needles[p])):
                nl += 1
                if found.get((p, i), set()) & train:
                    nh += 1
        print(f"split_v3 leakage {spn}: {nh}/{nl} ({pct(nh, nl)}) of long Sabche spans appear in a train book")
    cross = [(a, b) for a, b in pairs if a in split and b in split and split[a]["split"] != split[b]["split"]]
    print(f"linked pairs straddling split_v3 splits: {len(cross)}"
          + (f" e.g. {cross[:6]}" if cross else ""))
    outside = [(a, b) for a, b in pairs if (a in split) != (b in split)]
    print(f"linked pairs with one book outside split_v3: {len(outside)}")

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in book_rows:
        r["n_long_repeated"] = sum(1 for (p, _) in found if p == r["pecha_id"])
    with OUT_BOOKS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(book_rows[0]))
        w.writeheader(); w.writerows(book_rows)
    with (ROOT / "results/audit/sabche_links.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["a", "b", "share_a_in_b"])
        for (a, b) in sorted(links):
            w.writerow([a, b, round(share[(a, b)] / tot[a], 3)])
    print(f"\nwrote {OUT}, {OUT_BOOKS}, results/audit/sabche_links.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
