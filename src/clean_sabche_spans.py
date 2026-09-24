#!/usr/bin/env python3
"""
Build the cleaned Sabche span sidecar. Does not rewrite any Sabche.yml.

Usage:
    python src/clean_sabche_spans.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from check_boundary_snapping import BOUNDARY_CHARS  # noqa: E402
AUDIT = ROOT / "data/tsawa_audit.csv"
OUT = ROOT / "data/sabche_spans_clean.csv"
OUT_BOOKS = ROOT / "data/sabche_book_verdicts.csv"

MAX_WALK = 3
SWITCH_COST = 3.0
WINDOW = 50_000
MIN_GAIN = 0.20
MIN_HEADING = 0.60
EXCLUDE_BELOW = 0.30

ORD = ("དང་པོ", "གཉིས་པ", "གསུམ་པ", "བཞི་པ", "ལྔ་པ", "དྲུག་པ", "བདུན་པ",
       "བརྒྱད་པ", "དགུ་པ", "བཅུ་པ")
ANCHOR = re.compile("(?:" + "|".join(ORD) + ")[^།]{0,200}?ནི")
# new-batch outline headings are numbered (1) / {1} / [1] / ༡༽ instead
NUMBERED = re.compile(r"^\s*(?:[\(\{\[]\s*[0-9༠-༩]+\s*[\)\}\]]|[0-9༠-༩]+\s*[༽\.)])")
# ...or simply sit alone on a line: "\n<heading>།\n"
LINE_END = re.compile(r"^[\u0f0d\u0f0e \t\xa0]*(?:\n|$)")


def load_sabche(opf: Path) -> list[tuple[str, int, int]]:
    import yaml

    p = opf / "layers/v001/Sabche.yml"
    if not p.is_file():
        return []
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    anns = d.get("annotations") or {}
    items = anns.items() if isinstance(anns, dict) else enumerate(anns)
    out = []
    for aid, a in items:
        sp = (a or {}).get("span") or {}
        s, e = sp.get("start"), sp.get("end")
        if s is None or e is None or int(e) <= int(s):
            continue
        out.append((str(aid), int(s), int(e)))
    return sorted(out, key=lambda x: (x[1], x[2]))


def heading_like(text: str, s: int, e: int) -> bool:
    seg = text[max(0, s - 2): e]
    if any(o in seg[:14] for o in ORD):
        return True
    if "ནི" in text[max(s, e - 4): e + 2]:
        return True
    if NUMBERED.match(text[s:e]):
        return True
    line_start = s == 0 or text[s - 1] == "\n" or text[max(0, s - 3):s].lstrip("༈ ") == "" \
        or text[max(0, s - 3):s].endswith("\n༈ ")
    return line_start and bool(LINE_END.match(text[e:e + 6]))


def rate(text, spans) -> float:
    return sum(heading_like(text, s, e) for s, e in spans) / max(len(spans), 1)


def realign(text: str, spans: list[tuple[int, int]]) -> list[int]:
    """Return one offset per span (Viterbi over length-matched anchors)."""
    anchors = [(m.start(), m.end()) for m in ANCHOR.finditer(text)]
    by_len: dict[int, list[int]] = {}
    for a, b in anchors:
        by_len.setdefault(b - a, []).append(a)
    cands = []
    for s, e in spans:
        L = e - s
        c = set()
        for dl in (-1, 0, 1):
            for a in by_len.get(L + dl, ()):
                if abs(a - s) <= WINDOW:
                    c.add(a - s)
        cands.append(c)
    states = sorted({0} | set().union(*cands)) if cands else [0]
    idx = {o: i for i, o in enumerate(states)}
    S = len(states)
    INF = float("inf")
    cost = [0.0 if o in cands[0] else 1.0 for o in states]
    back = []
    for i in range(1, len(spans)):
        best_prev = min(range(S), key=lambda k: cost[k])
        bp_cost = cost[best_prev]
        new, bk = [INF] * S, [0] * S
        for k in range(S):
            stay, sw = cost[k], bp_cost + SWITCH_COST
            if stay <= sw:
                new[k], bk[k] = stay, k
            else:
                new[k], bk[k] = sw, best_prev
            new[k] += 0.0 if states[k] in cands[i] else 1.0
        back.append(bk)
        cost = new
    k = min(range(S), key=lambda j: cost[j])
    path = [k]
    for bk in reversed(back):
        k = bk[k]
        path.append(k)
    path.reverse()
    _ = idx
    return [states[k] for k in path]


def is_b(ch: str) -> bool:
    return ch in BOUNDARY_CHARS


def snap(text: str, s: int, e: int) -> tuple[int, int, str]:
    n = len(text)
    act = []
    if 0 < s < n and not is_b(text[s - 1]) and not is_b(text[s]):
        for k in range(1, MAX_WALK + 1):
            if s - k == 0 or is_b(text[s - k - 1]):
                s -= k
                act.append(f"start-{k}")
                break
        else:
            act.append("start_unfixed")
    if 0 < e < n and not is_b(text[e - 1]) and not is_b(text[e]):
        for k in range(1, MAX_WALK + 1):
            if e + k >= n or is_b(text[e + k]):
                e += k
                act.append(f"end+{k}")
                break
        else:
            act.append("end_unfixed")
    return s, e, "|".join(act)


def main() -> int:
    audit = list(csv.DictReader(AUDIT.open(encoding="utf-8")))
    rows, verdicts = [], []
    tot = Counter()
    for r in audit:
        pid = r["pecha_id"]
        raw = load_sabche(Path(r["opf_root"]))
        if not raw:
            continue
        batch = "old" if pid.startswith("P") else "new"
        text = Path(r["base_path"]).read_text(encoding="utf-8")
        spans = [(s, e) for _, s, e in raw]
        r0 = rate(text, spans)
        offs = [0] * len(spans)
        realigned = False
        r1 = r0
        if r0 < MIN_HEADING:
            cand = realign(text, spans)
            shifted = [(s + o, e + o) for (s, e), o in zip(spans, cand)]
            ok = all(0 <= s < e <= len(text) for s, e in shifted)
            r_try = rate(text, shifted) if ok else 0.0
            if ok and r_try >= MIN_HEADING and r_try - r0 >= MIN_GAIN:
                offs, realigned, r1 = cand, True, r_try
        excluded = r1 < EXCLUDE_BELOW
        n_seg = 1 + sum(1 for a, b in zip(offs, offs[1:]) if a != b)
        final = []
        for (aid, s0, e0), o in zip(raw, offs):
            s, e = s0 + o, e0 + o
            s, e, act = snap(text, s, e)
            bad = realigned and bool(act) and not heading_like(text, s, e)
            if not bad:
                final.append((s, e))
            actions = [a for a in (f"shift{o:+d}" if o else "", act) if a]
            rows.append({
                "pecha_id": pid, "batch": batch, "ann_id": aid,
                "raw_start": s0, "raw_end": e0, "start": s, "end": e,
                "action": "|".join(actions) or "keep",
                "dropped": excluded or bad,
                "reason": ("book_excluded_unrecoverable_offsets" if excluded
                           else "realigned_span_unverified" if bad else ""),
            })
            tot["spans"] += 1
            tot["dropped_unverified"] += bad
            tot["shifted"] += bool(o)
            tot["snapped"] += bool(act) and "unfixed" not in act
            tot["unfixed"] += "unfixed" in act
        # snapping can push two neighbours into each other (P000128: both
        # edges sat mid-syllable in the same word): trim the earlier span.
        book_rows = [x for x in rows if x["pecha_id"] == pid and not x["dropped"]]
        book_rows.sort(key=lambda x: (x["start"], x["end"]))
        for a, b in zip(book_rows, book_rows[1:]):
            if b["start"] < a["end"]:
                a["end"] = b["start"]
                a["action"] += "|trim_overlap"
                tot["trimmed"] += 1
        verdicts.append({
            "pecha_id": pid, "batch": batch, "n_spans": len(raw),
            "heading_rate_raw": round(r0, 3), "heading_rate_final": round(rate(text, final), 3),
            "realigned": realigned, "n_offset_segments": n_seg if realigned else 0,
            "verdict": "exclude" if excluded else ("realigned" if realigned else "keep"),
        })

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with OUT_BOOKS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(verdicts[0]))
        w.writeheader()
        w.writerows(verdicts)
    vc = Counter((v["batch"], v["verdict"]) for v in verdicts)
    print(f"books: {len(verdicts)}  " + "  ".join(f"{b}/{k}={n}" for (b, k), n in sorted(vc.items())))
    print(f"spans: {tot['spans']:,}  shifted {tot['shifted']:,}  snapped {tot['snapped']:,}  "
          f"left-dirty {tot['unfixed']:,}  dropped-unverified {tot['dropped_unverified']:,}  "
          f"overlap-trimmed {tot['trimmed']:,}")
    print(f"wrote {OUT}\nwrote {OUT_BOOKS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
