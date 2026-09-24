#!/usr/bin/env python3
"""
Check whether tsawa span start/end sit on Tibetan syllable boundaries.

Usage:
    python src/check_boundary_snapping.py \
        --audit-csv data/tsawa_audit.csv \
        --raw-opf-dir data/raw_opf \
        --out data/boundary_check.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

# Tsheg, white tsheg, shads, editorial brackets, ASCII + NBSP whitespace.
# ༼� Tsheg, white tsheg, shads, editorial brackets, ASCII + NBSP whitespace.
# ༼༽ / ༺༻ / []() stop walks so we do not absorb folio notes (e.g. ༽གཉིས་).
BOUNDARY_CHARS = set(
    "\u0f0b"  # tsheg
    "\u0f0c"  # white tsheg
    "\u0f0d\u0f0e\u0f0f\u0f10\u0f11"  # shad family
    "\u0f14"  # gter-shad ༔
    "\u0f3c\u0f3d\u0f3a\u0f3b"  # ༼ ༽ ༺ ༻
    "[]() \t\n\r\xa0"
)


def load_repos_with_tsawa(audit_csv: Path) -> list[dict]:
    """Read Phase 2 ``tsawa_audit.csv`` rows that have a non-empty Tsawa layer.

    Schema from ``audit_tsawa_data.py``: ``pecha_id``, ``has_tsawa_layer``,
    ``n_spans`` (not ``tsawa_span_count``), ``opf_root``, ``tsawa_path``,
    ``base_path``.
    """
    repos = []
    with open(audit_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_tsawa = row.get("has_tsawa_layer", "").strip().lower() == "true"
            span_count = int(row.get("n_spans") or 0)
            if has_tsawa and span_count > 0:
                repos.append(row)
    return repos


def infer_batch(pecha_id: str, audit_row: dict | None = None) -> str:
    """Same rule as ``audit_tsawa_data.infer_batch``: audit ``source_batch``,
    else ``P…`` → old, ``I…`` → new."""
    if audit_row and (audit_row.get("source_batch") or "").strip():
        return audit_row["source_batch"].strip()
    if pecha_id.startswith("P"):
        return "old"
    if pecha_id.startswith("I"):
        return "new"
    return "unknown"


def resolve_opf_root(raw_opf_dir: Path, repo_id: str, audit_row: dict | None = None) -> Path:
    """Git clones nest the Pecha tree as ``<ID>.opf/<ID>.opf/{base,layers}``."""
    if audit_row and audit_row.get("opf_root"):
        root = Path(audit_row["opf_root"])
        if (root / "base").is_dir():
            return root
    clone = raw_opf_dir / f"{repo_id}.opf"
    nested = clone / f"{repo_id}.opf"
    if (nested / "base").is_dir():
        return nested
    return clone


def load_spans(opf_dir: Path, tsawa_path: Path | None = None) -> list[dict]:
    """Load all tsawa spans for a single .opf repo, dropping zero-length ones."""
    if tsawa_path is None or not tsawa_path.is_file():
        tsawa_path = opf_dir / "layers" / "v001" / "Tsawa.yml"
    if not tsawa_path.exists():
        return []

    with open(tsawa_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    spans = []
    for ann_id, ann in (data.get("annotations") or {}).items():
        span = ann.get("span", {})
        start, end = span.get("start"), span.get("end")
        if start is None or end is None or start == end:
            continue
        spans.append({"ann_id": ann_id, "start": int(start), "end": int(end)})
    return spans


def _repr_char(ch: str) -> str:
    """Single-char field for CSV (escape newline so rows stay one line)."""
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    return ch


def check_span(text: str, start: int, end: int) -> dict:
    """Return cleanliness flags for one half-open [start, end) span."""
    n = len(text)
    start_is_clean = start == 0 or (start > 0 and text[start - 1] in BOUNDARY_CHARS)
    char_before = "" if start <= 0 else text[start - 1]
    if end <= 0 or end > n:
        end_is_clean = False
        last_char = ""
    else:
        last_char = text[end - 1]
        end_is_clean = last_char in BOUNDARY_CHARS
    return {
        "start_is_clean": start_is_clean,
        "char_before_start": _repr_char(char_before),
        "end_is_clean": end_is_clean,
        "last_char_of_span": _repr_char(last_char),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flag tsawa spans whose start/end fall mid-syllable (diagnostic)."
    )
    p.add_argument("--audit-csv", type=Path, required=True)
    p.add_argument("--raw-opf-dir", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="CSV of dirty spans only (repo_id, batch, ann_id, offsets, flags).",
    )
    p.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="If set, check these start/end instead of raw Tsawa.yml (e.g. snapped CSV).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repos = load_repos_with_tsawa(args.audit_csv)
    if not repos:
        raise SystemExit("No repos with tsawa spans found - check audit CSV column names.")

    sidecar_by_repo: dict[str, list[dict]] = {}
    if args.sidecar:
        sc_path = args.sidecar.expanduser().resolve()
        if not sc_path.is_file():
            raise SystemExit(f"Sidecar not found: {sc_path}")
        with sc_path.open(newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                if str(rec.get("dropped", "")).lower() == "true":
                    continue
                sidecar_by_repo.setdefault(rec["pecha_id"], []).append(
                    {
                        "ann_id": rec["ann_id"],
                        "start": int(rec["start"]),
                        "end": int(rec["end"]),
                    }
                )
        print(f"Using sidecar offsets from {sc_path}")

    dirty_rows: list[dict] = []
    stats = {
        name: {"total": 0, "dirty_start": 0, "dirty_end": 0, "dirty_either": 0}
        for name in ("old", "new", "combined", "unknown")
    }

    def bump(batch: str, key: str) -> None:
        stats[batch][key] += 1
        stats["combined"][key] += 1

    for row in repos:
        repo_id = row["pecha_id"]
        batch = infer_batch(repo_id, row)
        if batch not in stats:
            batch = "unknown"
        opf_dir = resolve_opf_root(args.raw_opf_dir, repo_id, row)
        base_path = Path(row["base_path"]) if row.get("base_path") else opf_dir / "base" / "v001.txt"
        tsawa_path = Path(row["tsawa_path"]) if row.get("tsawa_path") else None
        if not base_path.is_file():
            print(f"warning: missing base for {repo_id}: {base_path}")
            continue
        text = base_path.read_text(encoding="utf-8")
        if sidecar_by_repo:
            spans = sidecar_by_repo.get(repo_id, [])
        else:
            spans = load_spans(opf_dir, tsawa_path)
        for span in spans:
            bump(batch, "total")
            flags = check_span(text, span["start"], span["end"])
            dirty_s = not flags["start_is_clean"]
            dirty_e = not flags["end_is_clean"]
            if dirty_s:
                bump(batch, "dirty_start")
            if dirty_e:
                bump(batch, "dirty_end")
            if dirty_s or dirty_e:
                bump(batch, "dirty_either")
                dirty_rows.append(
                    {
                        "repo_id": repo_id,
                        "batch": batch,
                        "ann_id": span["ann_id"],
                        "start": span["start"],
                        "end": span["end"],
                        "start_is_clean": flags["start_is_clean"],
                        "char_before_start": flags["char_before_start"],
                        "end_is_clean": flags["end_is_clean"],
                        "last_char_of_span": flags["last_char_of_span"],
                    }
                )

    def cell(batch: str, key: str) -> str:
        n = stats[batch][key]
        tot = stats[batch]["total"]
        if key == "total":
            return f"{n}"
        if tot == 0:
            return f"{n} (n/a)"
        return f"{n} ({100.0 * n / tot:.1f}%)"

    print("=== BOUNDARY SNAPPING CHECK (by batch) ===")
    print(f"{'':22} {'old':>16} {'new':>16} {'combined':>16}")
    print(f"{'Total spans':22} {cell('old', 'total'):>16} {cell('new', 'total'):>16} {cell('combined', 'total'):>16}")
    print(f"{'Dirty start':22} {cell('old', 'dirty_start'):>16} {cell('new', 'dirty_start'):>16} {cell('combined', 'dirty_start'):>16}")
    print(f"{'Dirty end':22} {cell('old', 'dirty_end'):>16} {cell('new', 'dirty_end'):>16} {cell('combined', 'dirty_end'):>16}")
    print(f"{'Dirty either':22} {cell('old', 'dirty_either'):>16} {cell('new', 'dirty_either'):>16} {cell('combined', 'dirty_either'):>16}")
    if stats["unknown"]["total"]:
        print(f"(also {stats['unknown']['total']} spans with unknown batch)")

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "repo_id",
        "batch",
        "ann_id",
        "start",
        "end",
        "start_is_clean",
        "char_before_start",
        "end_is_clean",
        "last_char_of_span",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dirty_rows)
    print(f"Wrote {len(dirty_rows)} dirty spans to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
